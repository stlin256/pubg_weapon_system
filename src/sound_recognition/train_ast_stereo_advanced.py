import os
import argparse
import joblib
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import evaluate as hf_evaluate
from dataclasses import dataclass
from typing import Dict, List, Union

from . import config
from .dataset import get_data_splits, AudioDatasetStereoRaw

MODEL_PATH = "./ast_model"

@dataclass
class StereoDataCollator:
    feature_extractor: AutoFeatureExtractor
    def __call__(self, features: List[Dict[str, Union[np.ndarray, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 1. 从特征列表中分离原始音频和标签
        # 检查 `features` 的结构
        if not features:
            return {}

        # `features` 是一个列表, 其中的元素是 Dataset.__getitem__ 的输出
        # 我们期望的格式是 [{'raw_audio': ..., 'labels': ...}, ...]
        # 但看起来 Trainer 的默认 collate 行为改变了它
        # 让我们检查一下实际的结构
        
        # 调试输出
        # print("DEBUG: DataCollator received features:", features)
        
        raw_audios = [f["stereo_waveform"] for f in features]
        labels = [f["labels"] for f in features]

        # 2. 分别处理每个声道
        #    我们假设 raw_audios 中的每个元素都是 (2, n_samples)
        left_channel_audio = [audio[0, :] for audio in raw_audios]
        right_channel_audio = [audio[1, :] for audio in raw_audios]

        # 3. 使用 feature_extractor 独立处理每个声道
        left_inputs = self.feature_extractor(
            left_channel_audio, sampling_rate=config.SAMPLING_RATE,
            padding="max_length", return_tensors="pt"
        )
        right_inputs = self.feature_extractor(
            right_channel_audio, sampling_rate=config.SAMPLING_RATE,
            padding="max_length", return_tensors="pt"
        )
        
        # 4. 提取 input_values 并堆叠成 (batch, 2, height, width)
        #    left_inputs.input_values 的形状是 (batch, 1, height, width)
        #    我们需要将它们堆叠为 (batch, 2, height, width)
        left_spectrogram = left_inputs.input_values
        right_spectrogram = right_inputs.input_values
        
        # 确保两个声谱图的形状一致
        # (batch_size, 1, n_mels, n_frames)
        # 我们需要将它们合并为 (batch_size, 2, n_mels, n_frames)
        stacked_input_values = torch.cat([left_spectrogram, right_spectrogram], dim=1)
        
        # 5. 构建最终的输入字典
        final_inputs = {
            "input_values": stacked_input_values,
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        
        return final_inputs

def compute_metrics(eval_pred):
    accuracy_metric = hf_evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune an ADVANCED STEREO AST model.")
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    args = parser.parse_args()

    X_train_paths, _, y_train_df, _ = get_data_splits()
    y_train = y_train_df[args.target]
    
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        X_train_paths, y_train, test_size=config.VALIDATION_SIZE, 
        random_state=config.RANDOM_STATE, stratify=y_train
    )

    encoder = joblib.load(os.path.join(config.MODEL_SAVE_PATH, f'label_encoder_{args.target}.pkl'))
    y_train_encoded = encoder.transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    num_labels = len(encoder.classes_)

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    
    train_dataset = AudioDatasetStereoRaw(X_train_paths, y_train_encoded, config.SAMPLING_RATE)
    val_dataset = AudioDatasetStereoRaw(X_val_paths, y_val_encoded, config.SAMPLING_RATE)

    model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH, num_labels=num_labels, ignore_mismatched_sizes=True)
    
    original_layer = model.audio_spectrogram_transformer.embeddings.patch_embeddings.projection
    new_layer = torch.nn.Conv2d(
        in_channels=2, out_channels=original_layer.out_channels,
        kernel_size=original_layer.kernel_size, stride=original_layer.stride, padding=original_layer.padding
    )
    new_layer.weight.data.copy_(original_layer.weight.data.repeat(1, 2, 1, 1))
    new_layer.bias.data.copy_(original_layer.bias.data)
    model.audio_spectrogram_transformer.embeddings.patch_embeddings.projection = new_layer

    data_collator = StereoDataCollator(feature_extractor=feature_extractor)

    training_args = TrainingArguments(
        output_dir=os.path.join(config.MODEL_SAVE_PATH, f'ast_stereo_advanced_{args.target}_finetuned'),
        eval_strategy="epoch", save_strategy="epoch", learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs, weight_decay=0.01, load_best_model_at_end=True,
        remove_unused_columns=False, # 关键: 防止 Trainer 丢弃 'stereo_waveform' 列
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=data_collator, compute_metrics=compute_metrics
    )

    trainer.train()
    
    final_model_path = os.path.join(config.MODEL_SAVE_PATH, f'ast_stereo_advanced_{args.target}_final')
    trainer.save_model(final_model_path)

if __name__ == '__main__':
    main()