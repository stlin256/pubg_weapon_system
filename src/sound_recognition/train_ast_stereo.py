import os
import argparse
import joblib
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import evaluate as hf_evaluate

from . import config
from .dataset import get_data_splits, AudioDatasetStereo

MODEL_PATH = "./ast_model"

def compute_metrics(eval_pred):
    accuracy_metric = hf_evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a STEREO AST model.")
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
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
    
    train_dataset = AudioDatasetStereo(X_train_paths, y_train_encoded, feature_extractor, config.SAMPLING_RATE)
    val_dataset = AudioDatasetStereo(X_val_paths, y_val_encoded, feature_extractor, config.SAMPLING_RATE)

    model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH, num_labels=num_labels, ignore_mismatched_sizes=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(config.MODEL_SAVE_PATH, f'ast_stereo_{args.target}_finetuned'),
        eval_strategy="epoch", save_strategy="epoch", learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs, weight_decay=0.01, load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, tokenizer=feature_extractor, compute_metrics=compute_metrics,
    )

    trainer.train()
    
    final_model_path = os.path.join(config.MODEL_SAVE_PATH, f'ast_stereo_{args.target}_final')
    trainer.save_model(final_model_path)

if __name__ == '__main__':
    main()