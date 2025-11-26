import os
import numpy as np
import librosa
from tqdm import tqdm
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from . import config
from . import dataset

# --- 特征处理参数 ---
MAX_PADDING_LENGTH = 174 

def extract_features(file_path):
    """
    从单个音频文件中提取MFCC特征.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=config.SAMPLING_RATE, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=config.N_MFCC,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        
        # 对特征进行填充或截断
        if mfccs.shape[1] > MAX_PADDING_LENGTH:
            mfccs = mfccs[:, :MAX_PADDING_LENGTH]
        else:
            pad_width = MAX_PADDING_LENGTH - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_and_save_data(filepaths, labels_df, data_type):
    """
    处理整个数据集(训练集或测试集)的特征和标签, 并进行保存.
    
    Args:
        filepaths (list): 音频文件路径列表.
        labels_df (pd.DataFrame): 包含多个标签列的DataFrame.
        data_type (str): 'train' 或 'test', 用于文件名.
    """
    features = []
    
    # 使用 tqdm 创建一个进度条
    for file_path in tqdm(filepaths, desc=f"Processing features for {data_type} data"):
        mfccs = extract_features(file_path)
        if mfccs is not None:
            features.append(mfccs)
    
    # --- 保存特征 ---
    features = np.array(features)
    features_save_path = os.path.join(config.FEATURES_PATH, f'{data_type}_features.npy')
    np.save(features_save_path, features)
    print(f"Saved {data_type} features to {features_save_path}")

    # --- 对每个标签列进行编码并保存 ---
    for column in labels_df.columns:
        label_series = labels_df[column]
        
        # 创建并拟合/转换 LabelEncoder
        encoder = LabelEncoder()
        
        if data_type == 'train':
            # 如果是训练集, 需要 fit_transform
            encoded_labels = encoder.fit_transform(label_series)
            # 保存这个拟合好的 encoder
            encoder_path = os.path.join(config.MODEL_SAVE_PATH, f'label_encoder_{column}.pkl')
            joblib.dump(encoder, encoder_path)
            print(f"Label encoder for '{column}' saved to {encoder_path}")
        else:
            # 如果是测试集, 需要加载已有的 encoder 进行 transform
            encoder_path = os.path.join(config.MODEL_SAVE_PATH, f'label_encoder_{column}.pkl')
            encoder = joblib.load(encoder_path)
            encoded_labels = encoder.transform(label_series)

        # 保存编码后的标签
        labels_save_path = os.path.join(config.FEATURES_PATH, f'{data_type}_labels_{column}.npy')
        np.save(labels_save_path, encoded_labels)
        print(f"Saved encoded '{column}' labels for {data_type} to {labels_save_path}")


def main():
    """
    主执行函数.
    """
    print("--- Starting feature and multi-label extraction ---")
    
    # 1. 加载数据路径和多标签DataFrame
    X_train_paths, X_test_paths, y_train_df, y_test_df = dataset.get_data_splits()
    
    # 2. 处理并保存训练集数据
    if X_train_paths:
        process_and_save_data(X_train_paths, y_train_df, 'train')
    
    # 3. 处理并保存测试集数据
    if X_test_paths:
        process_and_save_data(X_test_paths, y_test_df, 'test')
    
    print("\n--- Feature extraction complete ---")

if __name__ == '__main__':
    main()