import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from hear21passt.base import get_basic_model
from torch import nn

from . import config
from .dataset import AudioDataset, AudioDatasetPaSST, get_data_splits
from .train_passt import StereoDataCollator


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, target_name):
    """
    绘制并保存混淆矩阵.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {model_name} on target "{target_name}"')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_path = os.path.join(config.FIGURES_PATH, f'cm_{model_name}_{target_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix for '{target_name}' saved to {save_path}")

def evaluate_sklearn_model(model, model_name, target, X_test, y_test_true):
    """
    评估单个目标的 scikit-learn 模型.
    """
    print(f"\n--- Evaluating scikit-learn model: {model_name} for target '{target}' ---")
    encoder = joblib.load(os.path.join(config.MODEL_SAVE_PATH, f'label_encoder_{target}.pkl'))
    y_true_encoded = encoder.transform(y_test_true)
    class_names = encoder.classes_
    
    y_pred = model.predict(X_test)
    
    report = classification_report(y_true_encoded, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    print(f"Classification Report for '{target}':\n{pd.DataFrame(report).transpose()}")
    plot_confusion_matrix(y_true_encoded, y_pred, class_names, model_name, target)
    
    macro_avg = report['macro avg']
    return {
        'model': model_name, 'target': target, 'accuracy': report['accuracy'],
        'precision (macro)': macro_avg['precision'], 'recall (macro)': macro_avg['recall'],
        'f1-score (macro)': macro_avg['f1-score']
    }

def evaluate_multi_target_model(model, model_name, X_test, y_test_dict):
    """
    评估多输出 scikit-learn 模型.
    """
    print(f"\n--- Evaluating multi-target model: {model_name} ---")
    y_pred_multi = model.predict(X_test)
    results = []
    for i, target in enumerate(['weapon', 'distance', 'direction']):
        encoder = joblib.load(os.path.join(config.MODEL_SAVE_PATH, f'label_encoder_{target}.pkl'))
        y_true_encoded = encoder.transform(y_test_dict[target])
        class_names = encoder.classes_
        
        y_pred = y_pred_multi[:, i]
        
        report = classification_report(y_true_encoded, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        print(f"\nClassification Report for sub-task '{target}':\n{pd.DataFrame(report).transpose()}")
        plot_confusion_matrix(y_true_encoded, y_pred, class_names, model_name, target)
        
        macro_avg = report['macro avg']
        results.append({
            'model': model_name, 'target': target, 'accuracy': report['accuracy'],
            'precision (macro)': macro_avg['precision'], 'recall (macro)': macro_avg['recall'],
            'f1-score (macro)': macro_avg['f1-score']
        })
    return results

def evaluate_ast_model(model_path, target, X_test_paths, y_test_true, use_stereo=False):
    """
    评估一个微调过的 AST 模型.
    """
    model_name = os.path.basename(model_path).replace('_final', '')
    print(f"\n--- Evaluating AST model: {model_name} for target '{target}' (Stereo: {use_stereo}) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path).to(device)
    encoder = joblib.load(os.path.join(config.MODEL_SAVE_PATH, f'label_encoder_{target}.pkl'))
    class_names = encoder.classes_

    if use_stereo:
        from .dataset import AudioDatasetStereo
        test_dataset = AudioDatasetStereo(X_test_paths, y_test_true, feature_extractor, config.SAMPLING_RATE)
    else:
        test_dataset = AudioDataset(X_test_paths, y_test_true, feature_extractor, config.SAMPLING_RATE)
        
    test_loader = DataLoader(test_dataset, batch_size=16)

    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            inputs = batch['input_values'].to(device)
            logits = model(inputs).logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

    report = classification_report(y_test_true, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    print(f"Classification Report for '{target}':\n{pd.DataFrame(report).transpose()}")
    plot_confusion_matrix(y_test_true, all_preds, class_names, model_name, target)
    
    macro_avg = report['macro avg']
    return {
        'model': model_name, 'target': target, 'accuracy': report['accuracy'],
        'precision (macro)': macro_avg['precision'], 'recall (macro)': macro_avg['recall'],
        'f1-score (macro)': macro_avg['f1-score']
    }

def evaluate_passt_model(model_path, target, X_test_paths, y_test_true):
    """
    Evaluates a fine-tuned PaSST model based on the new training script.
    """
    model_name = os.path.basename(model_path).replace('_final', '')
    print(f"\n--- Evaluating PaSST model: {model_name} for target '{target}' ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load class names from the classes.txt file
    with open(os.path.join(model_path, "classes.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    num_labels = len(class_names)

    # --- 1. Reconstruct the model architecture ---
    model = get_basic_model(mode="logits")

    # Adapt Patch Embedding for Stereo Input
    old_conv = model.net.patch_embed.proj
    new_conv = nn.Conv2d(
        in_channels=2, 
        out_channels=old_conv.out_channels, 
        kernel_size=old_conv.kernel_size, 
        stride=old_conv.stride, 
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    model.net.patch_embed.proj = new_conv

    # Adapt Classification Head
    old_head = model.net.head[1]
    model.net.head[1] = nn.Linear(old_head.in_features, num_labels)
    
    # --- 2. Load the fine-tuned weights ---
    state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
    model.net.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 3. Create Dataset and DataLoader ---
    test_dataset = AudioDatasetPaSST(X_test_paths, y_test_true, target_sampling_rate=32000)
    data_collator = StereoDataCollator()
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator, shuffle=False)
    
    # --- 4. Run Inference ---
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            inputs = batch['input_values'].to(device)
            
            # --- Replicate the forward pass from training ---
            x_left, x_right = inputs[:, 0], inputs[:, 1]
            spec_left = model.mel(x_left).unsqueeze(1)
            spec_right = model.mel(x_right).unsqueeze(1)
            x = torch.cat([spec_left, spec_right], dim=1)
            
            logits = model.net(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            # --- End of replication ---

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

    # --- 5. Report and Return Results ---
    report = classification_report(y_test_true, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    print(f"Classification Report for '{target}':\n{pd.DataFrame(report).transpose()}")
    plot_confusion_matrix(y_test_true, all_preds, class_names, model_name, target)
    
    macro_avg = report['macro avg']
    return {
        'model': model_name, 'target': target, 'accuracy': report['accuracy'],
        'precision (macro)': macro_avg['precision'], 'recall (macro)': macro_avg['recall'],
        'f1-score (macro)': macro_avg['f1-score']
    }


import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument('--model_type', type=str, choices=['passt', 'ast', 'sklearn'],
                        help="Specify a model type to evaluate (e.g., 'passt'). If not provided, all models are evaluated.")
    args = parser.parse_args()

    print("--- Starting comprehensive model evaluation ---")
    if args.model_type:
        print(f"--- Mode: Evaluating ONLY '{args.model_type}' models ---")

    X_test_mfcc = np.load(os.path.join(config.FEATURES_PATH, 'test_features.npy'))
    nsamples, nx, ny = X_test_mfcc.shape
    X_test_reshaped = X_test_mfcc.reshape((nsamples, nx * ny))
    
    _, X_test_paths, _, y_test_df = get_data_splits()

    y_test_dict = {
        target: y_test_df[target].values for target in ['weapon', 'distance', 'direction']
    }
    
    all_results = []

    # --- Evaluate scikit-learn models ---
    sklearn_model_paths = []
    if not args.model_type or args.model_type == 'sklearn':
        sklearn_model_paths = glob.glob(os.path.join(config.MODEL_SAVE_PATH, '*.pkl'))
        sklearn_model_paths = [p for p in sklearn_model_paths if 'label_encoder' not in os.path.basename(p)]
    
    for model_path in sklearn_model_paths:
        model_filename = os.path.basename(model_path)
        model_name = model_filename.replace('.pkl', '')
        model = joblib.load(model_path)
        
        if '_all' in model_name:
            results = evaluate_multi_target_model(model, model_name, X_test_reshaped, y_test_dict)
            all_results.extend(results)
        else:
            try:
                target = model_name.split('_')[1]
                if target in y_test_dict:
                    result = evaluate_sklearn_model(model, model_name, target, X_test_reshaped, y_test_dict[target])
                    all_results.append(result)
            except IndexError:
                print(f"Warning: Could not parse target for sklearn model '{model_name}'.")

    # --- Evaluate Transformer models ---
    transformer_model_paths = []
    if not args.model_type or args.model_type in ['ast', 'passt']:
        all_model_dirs = glob.glob(os.path.join(config.MODEL_SAVE_PATH, 'ast_*')) + \
                         glob.glob(os.path.join(config.MODEL_SAVE_PATH, 'passt_*'))
        
        # 过滤掉 checkpoints 和 finetuned 临时目录
        transformer_model_paths = [
            p for p in all_model_dirs
            if os.path.isdir(p) and 'finetuned' not in p and 'checkpoints' not in p
        ]
    
    for model_path in transformer_model_paths:
        model_basename = os.path.basename(model_path)
        try:
            # Check for PaSST models
            if 'passt' in model_basename and (not args.model_type or args.model_type == 'passt'):
                target = model_basename.split('_')[1]
                if target in y_test_dict:
                    encoder = joblib.load(os.path.join(config.MODEL_SAVE_PATH, f'label_encoder_{target}.pkl'))
                    y_true_encoded = encoder.transform(y_test_dict[target])
                    result = evaluate_passt_model(model_path, target, X_test_paths, y_true_encoded)
                    if result: all_results.append(result)
            # Handle AST models
            elif 'ast' in model_basename and (not args.model_type or args.model_type == 'ast'):
                is_stereo = 'stereo' in model_basename
                target = model_basename.split('_')[2 if is_stereo else 1]
                if target in y_test_dict:
                    encoder = joblib.load(os.path.join(config.MODEL_SAVE_PATH, f'label_encoder_{target}.pkl'))
                    y_true_encoded = encoder.transform(y_test_dict[target])
                    result = evaluate_ast_model(model_path, target, X_test_paths, y_true_encoded, use_stereo=is_stereo)
                    all_results.append(result)
        except IndexError:
            print(f"Warning: Could not parse target for Transformer model '{model_basename}'.")

    # --- Save final report ---
    if all_results:
        results_df = pd.DataFrame(all_results).sort_values(by=['target', 'accuracy'], ascending=[True, False])
        report_path = os.path.join(config.REPORTS_PATH, 'evaluation_results.csv')
        results_df.to_csv(report_path, index=False)
        
        print("\n--- Evaluation Complete ---")
        print("Final Evaluation Report:")
        print(results_df)
        print(f"\nFull report saved to {report_path}")

if __name__ == '__main__':
    main()