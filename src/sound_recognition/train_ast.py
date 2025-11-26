import os
import argparse
import joblib
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
# [修改1] 移除 evaluate 库，改用本地 sklearn，防止联网下载导致卡死
from sklearn.metrics import accuracy_score

from . import config
from .dataset import get_data_splits, AudioDataset

# --- 本地模型路径 ---
MODEL_PATH = "./ast_model"

def compute_metrics(eval_pred):
    """
    计算评估指标 (修复版)
    """
    predictions = eval_pred.predictions
    
    # [修改2] AST 模型可能会返回 (logits, hidden_states) 的元组
    # 如果不解包，np.argmax 会报错或导致评估进程卡死
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    predictions = np.argmax(predictions, axis=1)
    
    # [修改1] 使用本地 sklearn 计算准确率
    acc = accuracy_score(eval_pred.label_ids, predictions)
    return {"accuracy": acc}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Audio Spectrogram Transformer (AST) model.")
    parser.add_argument('--target', type=str, required=True, help="The target to predict (e.g., 'weapon', 'distance', 'direction').")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate.")
    args = parser.parse_args()

    print(f"--- Fine-tuning AST model for target '{args.target}' ---")
    
    # --- 1. 加载数据 ---
    X_train_paths, X_test_paths, y_train_df, y_test_df = get_data_splits()
    
    # 根据 target 获取对应的标签
    y_train = y_train_df[args.target]
    
    # 进一步从训练集中划分出一个验证集
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        X_train_paths, y_train, 
        test_size=config.VALIDATION_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y_train
    )

    # 对标签进行编码
    encoder = joblib.load(os.path.join(config.MODEL_SAVE_PATH, f'label_encoder_{args.target}.pkl'))
    y_train_encoded = encoder.transform(y_train)
    y_val_encoded = encoder.transform(y_val)

    num_labels = len(encoder.classes_)
    print(f"Number of unique labels for '{args.target}': {num_labels}")

    # --- 2. 加载 Feature Extractor 和创建 Dataset ---
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    
    train_dataset = AudioDataset(X_train_paths, y_train_encoded, feature_extractor, config.SAMPLING_RATE)
    val_dataset = AudioDataset(X_val_paths, y_val_encoded, feature_extractor, config.SAMPLING_RATE)

    # --- 3. 加载预训练模型 ---
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_PATH,
        num_labels=num_labels,
        ignore_mismatched_sizes=True 
    )

    # --- 4. 定义训练参数 ---
    output_dir = os.path.join(config.MODEL_SAVE_PATH, f'ast_{args.target}_finetuned')
    
    # [修改3] 兼容新旧版本的 evaluate 参数名
    strategy_arg = "eval_strategy" if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames else "evaluation_strategy"

    training_args = TrainingArguments(
        output_dir=output_dir,
        **{strategy_arg: "epoch"}, # 动态使用正确的参数名
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=None,         # [配置] 硬盘充足，不限制保存数量
        dataloader_num_workers=4,      # [配置] 加速数据加载
        fp16=False,                    # [配置] 显存充足，使用 FP32 全精度以获得最佳效果
        push_to_hub=False
    )

    # --- 5. 初始化 Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor, 
        compute_metrics=compute_metrics,
    )

    # --- 6. 开始训练 ---
    print("Starting AST model fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")
    
    # --- 7. 保存最终模型 ---
    final_model_path = os.path.join(config.MODEL_SAVE_PATH, f'ast_{args.target}')
    trainer.save_model(final_model_path)
    
    # [新增] 保存类别名称，方便推理时使用
    with open(os.path.join(final_model_path, "classes.txt"), "w") as f:
        for cls in encoder.classes_:
            f.write(f"{cls}\n")
            
    print(f"Final fine-tuned AST model saved to {final_model_path}")


if __name__ == '__main__':
    main()