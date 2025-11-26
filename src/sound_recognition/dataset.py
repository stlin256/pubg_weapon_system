import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config

def parse_info_from_filename(filename):
    """
    从文件名中解析出武器、距离和方向信息.
    例如: 'ak_0m_center_0017.mp3' -> ('ak', '0m', 'center')
    对于 'nogun_None_None_2132.mp3' 这类文件, 返回 ('nogun', 'None', 'None')
    """
    parts = os.path.basename(filename).split('_')
    if len(parts) >= 3:
        weapon = parts[0]
        distance = parts[1]
        direction = parts[2]
        return weapon, distance, direction
    else:
        # 返回None或默认值, 以便后续可以过滤掉这些无效数据
        return None, None, None

def load_audio_paths_and_labels(data_path):
    """
    遍历数据目录, 加载所有音频文件的路径并从文件名解析多个标签.
    """
    filepaths = []
    labels = {
        'weapon': [],
        'distance': [],
        'direction': []
    }
    
    # 遍历 train 和 test 文件夹
    for split in ['gun_sound_train', 'gun_sound_test']:
        split_path = os.path.join(data_path, split)
        
        if not os.path.isdir(split_path):
            print(f"Warning: Directory not found at '{split_path}'. Skipping.")
            continue
            
        search_pattern = os.path.join(split_path, '*.mp*')
        audio_files = glob.glob(search_pattern)
        
        for audio_file in audio_files:
            weapon, distance, direction = parse_info_from_filename(audio_file)
            if weapon is not None:
                filepaths.append(audio_file)
                labels['weapon'].append(weapon)
                labels['distance'].append(distance)
                labels['direction'].append(direction)
    
    print(f"Total valid audio files found: {len(filepaths)}")
    # 使用 pandas 创建一个 DataFrame 以便更好地展示和分析标签
    labels_df = pd.DataFrame(labels)
    print("Label distribution preview:")
    print(labels_df.apply(lambda col: col.value_counts()).fillna(0).astype(int))
    
    return filepaths, labels_df


def get_data_splits():
    """
    加载数据并划分为训练集和测试集.
    返回文件路径列表和一个包含多个标签列的DataFrame.
    """
    filepaths, labels_df = load_audio_paths_and_labels(config.AUDIO_DATA_PATH)
    
    if not filepaths:
        print("Error: No audio files found. Cannot create data splits.")
        return [], [], pd.DataFrame(), pd.DataFrame()

    # 我们使用 'weapon' 标签进行分层抽样, 因为它是最重要的目标
    # stratify=labels_df['weapon']
    X_train, X_test, y_train_df, y_test_df = train_test_split(
        filepaths,
        labels_df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels_df['weapon']
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train_df, y_test_df

if __name__ == '__main__':
    print("--- Testing dataset loading and splitting ---")
    X_train, X_test, y_train_df, y_test_df = get_data_splits()
    if X_train:
        print("\n--- Training Data Sample ---")
        print("File path:", X_train[0])
        print("Labels:\n", y_train_df.iloc[0])
        
        print("\n--- Test Data Sample ---")
        print("File path:", X_test[0])
        print("Labels:\n", y_test_df.iloc[0])

# --- PyTorch Dataset for AST ---
import torch
import librosa
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, feature_extractor, target_sampling_rate):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.target_sampling_rate = target_sampling_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # 加载音频文件
        speech, sr = librosa.load(file_path, sr=self.target_sampling_rate)
        
        # 使用 feature_extractor 处理音频
        # padding="max_length" 会自动填充或截断到模型期望的长度
        inputs = self.feature_extractor(speech, sampling_rate=self.target_sampling_rate, padding="max_length", return_tensors="pt")
        
        # feature_extractor 的输出是一个字典, 我们需要提取 input_values
        # .squeeze(0) 是因为 feature_extractor 默认会添加一个 batch 维度
        input_values = inputs.input_values.squeeze(0)
        
        # 获取对应的标签
        label = self.labels[idx]
        
        return {"input_values": input_values, "labels": torch.tensor(label, dtype=torch.long)}

import numpy as np
# --- PyTorch Dataset for Stereo AST ---
class AudioDatasetStereo(Dataset):
    def __init__(self, file_paths, labels, feature_extractor, target_sampling_rate):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.target_sampling_rate = target_sampling_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        speech, sr = librosa.load(self.file_paths[idx], sr=self.target_sampling_rate, mono=False)
        if speech.ndim == 1:
            speech = np.stack([speech, speech])
        speech = np.mean(speech, axis=0)
        inputs = self.feature_extractor(speech, sampling_rate=self.target_sampling_rate, padding="max_length", return_tensors="pt")
        return {"input_values": inputs.input_values.squeeze(0), "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

# --- PyTorch Dataset for Raw Stereo AST ---
class AudioDatasetStereo(Dataset):
    def __init__(self, file_paths, labels, feature_extractor, target_sampling_rate):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.target_sampling_rate = target_sampling_rate
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        speech, sr = librosa.load(self.file_paths[idx], sr=self.target_sampling_rate, mono=False)
        if speech.ndim == 1:
            speech = np.stack([speech, speech])
        speech = np.mean(speech, axis=0)
        inputs = self.feature_extractor(speech, sampling_rate=self.target_sampling_rate, padding="max_length", return_tensors="pt")
        return {"input_values": inputs.input_values.squeeze(0), "labels": torch.tensor(self.labels[idx], dtype=torch.long)}
# --- PyTorch Dataset for PaSST Stereo Audio ---
class AudioDatasetPaSST(Dataset):
    def __init__(self, file_paths, labels, target_sampling_rate):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sampling_rate = target_sampling_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        speech, sr = librosa.load(self.file_paths[idx], sr=self.target_sampling_rate, mono=False)
        
        # Ensure the audio is stereo
        if speech.ndim == 1:
            speech = np.stack([speech, speech])
        
        # Ensure it has 2 channels
        if speech.shape[0] != 2 and speech.shape[1] > 2: # A quick check to avoid misinterpreting shape
             speech = speech.T

        if speech.shape[0] != 2:
            # If still not stereo, duplicate the first channel
            speech = np.stack([speech[0], speech[0]])

        label = self.labels[idx]
        return {"input_values": speech, "labels": torch.tensor(label, dtype=torch.long)}
