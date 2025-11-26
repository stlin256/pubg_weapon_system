import torch
import torch.nn as nn
import argparse
import numpy as np
import joblib
import os
import warnings
from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
from hear21passt.base import get_basic_model

# 假设这些是你项目中的本地模块
from .dataset import get_data_splits, AudioDatasetPaSST
from . import config

# 过滤掉 PaSST 的尺寸警告，让控制台更清爽
warnings.filterwarnings("ignore", message="Input image size", category=UserWarning)

def compute_metrics(p):
    """Computes accuracy metric."""
    preds = np.argmax(p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

class StereoDataCollator:
    """
    A robust collator that handles dynamic padding for stereo audio waveforms
    AND enforces a maximum duration to prevent position embedding errors.
    """
    def __call__(self, features):
        input_values = [torch.from_numpy(f["input_values"].astype(np.float32)) for f in features]
        labels = [f["labels"] for f in features]

        # PaSST (AudioSet) max duration is ~10s. 
        # Sampling rate is 32000, so max samples = 320,000.
        MAX_SAMPLES = 32000 * 10 
        
        truncated_inputs = []
        for wav in input_values:
            # wav shape is [Channels, Time]
            if wav.shape[1] > MAX_SAMPLES:
                wav = wav[:, :MAX_SAMPLES]
            truncated_inputs.append(wav)
        
        input_values = truncated_inputs

        max_len = max(wav.shape[1] for wav in input_values)

        padded_inputs = []
        for wav in input_values:
            pad_needed = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, pad_needed), "constant", 0)
            padded_inputs.append(padded_wav)

        batched_inputs = torch.stack(padded_inputs)
        batched_labels = torch.stack(labels)

        return {
            "input_values": batched_inputs,
            "labels": batched_labels
        }

class PaSSTForTraining(nn.Module):
    """
    A wrapper for the PaSST model to make it compatible with the Hugging Face Trainer.
    """
    def __init__(self, passt_model, num_labels):
        super().__init__()
        self.mel = passt_model.mel
        self.net = passt_model.net
        
        # Replace Head
        old_head = self.net.head[1]
        self.net.head[1] = nn.Linear(old_head.in_features, num_labels)
        
        self.loss_fct = CrossEntropyLoss()

        # --- 修复 AttributeError: 'PaSSTForTraining' object has no attribute '_keys_to_ignore_on_save' ---
        # HuggingFace Trainer 需要这个属性存在，即使是 None
        self._keys_to_ignore_on_save = None
        # ---------------------------------------------------------------------------------------------

    def forward(self, input_values, labels=None, **kwargs):
        """
        input_values: [Batch, 2, Time]
        """
        device = input_values.device
        
        # 1. Split Stereo
        x_left = input_values[:, 0]  # [Batch, Time]
        x_right = input_values[:, 1] # [Batch, Time]
        
        # 2. Extract Mel Spectrograms
        spec_left = self.mel(x_left)
        spec_right = self.mel(x_right)
        
        # Ensure 4D Shape [Batch, 1, Freq, Time]
        if spec_left.dim() == 3:
            spec_left = spec_left.unsqueeze(1)
        if spec_right.dim() == 3:
            spec_right = spec_right.unsqueeze(1)

        # 3. Stack on Channel Dimension (dim=1)
        x = torch.cat([spec_left, spec_right], dim=1)
        
        # 4. Pass to Transformer
        logits = self.net(x)
        
        # Handle Tuple Output (Class Token, Distillation Token)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.to(device))

        return (loss, logits) if loss is not None else logits

    def state_dict(self, *args, **kwargs):
        return self.net.state_dict(*args, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PaSST model for audio classification.")
    parser.add_argument('--target', type=str, required=True, choices=['weapon', 'distance', 'direction'], help="The target label to train on.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    args = parser.parse_args()

    print("--- 1. Loading and splitting data ---")
    X_train, X_test, y_train_df, y_test_df = get_data_splits()

    if not X_train:
        print("Error: No training data found. Exiting.")
        return

    print("--- 2. Preprocessing data ---")
    encoder_path = os.path.join(config.MODEL_SAVE_PATH, f'label_encoder_{args.target}.pkl')
    label_encoder = joblib.load(encoder_path)
    y_train = label_encoder.transform(y_train_df[args.target])
    y_test = label_encoder.transform(y_test_df[args.target])
    
    num_labels = len(label_encoder.classes_)
    print(f"Number of unique labels for '{args.target}': {num_labels}")

    print("--- 3. Loading and adapting PaSST model ---")
    model = get_basic_model(mode="logits")

    # --- Modify Patch Embedding for Stereo Input ---
    old_conv = model.net.patch_embed.proj
    new_conv = nn.Conv2d(
        in_channels=2, 
        out_channels=old_conv.out_channels, 
        kernel_size=old_conv.kernel_size, 
        stride=old_conv.stride, 
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )

    with torch.no_grad():
        original_weights = old_conv.weight 
        new_weights = torch.cat([original_weights, original_weights], dim=1)
        new_conv.weight = nn.Parameter(new_weights / 2.0)
        if old_conv.bias is not None:
            new_conv.bias = old_conv.bias

    model.net.patch_embed.proj = new_conv
    print("Adapted PaSST model for stereo input.")
    
    training_model = PaSSTForTraining(model, num_labels)

    print("--- 4. Creating datasets ---")
    train_dataset = AudioDatasetPaSST(X_train, y_train, target_sampling_rate=32000)
    test_dataset = AudioDatasetPaSST(X_test, y_test, target_sampling_rate=32000)
    
    data_collator = StereoDataCollator()

    print("--- 5. Defining training arguments ---")
    strategy_arg = "eval_strategy" if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames else "evaluation_strategy"
    
    args_dict = {
        # 临时保存目录 (Checkpoints)
        "output_dir": f"./trained_models/passt_{args.target}_checkpoints",
        strategy_arg: "epoch",
        "save_strategy": "epoch",
        "learning_rate": 3e-5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_dir": './logs',
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "remove_unused_columns": False,
        "save_total_limit": 2,
        "dataloader_num_workers": 4,
        "fp16": torch.cuda.is_available(),
    }
    
    training_args = TrainingArguments(**args_dict)

    print("--- 6. Initializing Trainer ---")
    trainer = Trainer(
        model=training_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("--- 7. Starting training ---")
    trainer.train()
    
    print("--- 8. Saving the fine-tuned model ---")
    # 确保最终保存的模型名称包含 passt 和 target
    final_model_dir = f"./trained_models/passt_{args.target}"
    os.makedirs(final_model_dir, exist_ok=True)
    
    final_model_path = os.path.join(final_model_dir, "pytorch_model.bin")
    
    # 保存模型权重 (调用自定义的 state_dict 方法，只保存 net 部分)
    torch.save(training_model.net.state_dict(), final_model_path)
    
    # 同时也保存一下 Label Encoder 的类名，方便推理时使用
    with open(os.path.join(final_model_dir, "classes.txt"), "w") as f:
        for cls in label_encoder.classes_:
            f.write(f"{cls}\n")

    print(f"Success! Model saved to: {final_model_path}")

if __name__ == '__main__':
    main()