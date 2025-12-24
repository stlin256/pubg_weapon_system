#用于生成梅尔谱图可视化
import os
import librosa
import librosa.display
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor
from hear21passt.base import get_basic_model
import warnings

# --- 忽略警告 ---
warnings.filterwarnings("ignore")

# --- 配置 ---
AUDIO_FILE_PATH = "sounds/gun_sound_train/m4_200m_right_1412.mp3"
AST_MODEL_PATH = "./ast_model"
OUTPUT_DIR = "imgs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 绘图辅助函数

def save_single_plot(spectrogram, sr, title, filename, x_limit=None):
    """保存单张图片 (用于 AST)"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    
    if x_limit is not None:
        plt.xlim(0, x_limit)
        
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved AST plot: {output_path}")

def save_dual_channel_plot(spec_left, spec_right, sr, title_main, filename, x_limit=None):
    """保存双声道上下并排的图片 (用于 PaSST)"""
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    
    # 左声道
    img_l = librosa.display.specshow(spec_left, sr=sr, x_axis='time', y_axis='mel', ax=axs[0])
    axs[0].set_title(f"{title_main} - Left Channel")
    fig.colorbar(img_l, ax=axs[0], format='%+2.0f dB')

    # 右声道
    img_r = librosa.display.specshow(spec_right, sr=sr, x_axis='time', y_axis='mel', ax=axs[1])
    axs[1].set_title(f"{title_main} - Right Channel")
    fig.colorbar(img_r, ax=axs[1], format='%+2.0f dB')

    if x_limit is not None:
        axs[0].set_xlim(0, x_limit)
        axs[1].set_xlim(0, x_limit)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PaSST plot: {output_path}")

# 1. AST 模型处理 (显式混合双声道)
def generate_ast_images(audio_path, model_path):
    print("--- Processing AST ---")
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        sampling_rate = feature_extractor.sampling_rate
        
        # 1. 先加载原始立体声 (mono=False)
        speech, sr = librosa.load(audio_path, sr=sampling_rate, mono=False)
        
        # 2. 显式混合声道
        if speech.ndim > 1:
            print("AST Info: Mixing stereo channels to mono (Average).")
            # axis=0 表示沿着声道维度求平均 ( Left + Right ) / 2
            speech = np.mean(speech, axis=0) 
        else:
            print("AST Info: Audio is already mono.")

        # --- 图1: AST Clean Original (自然长度) ---
        inputs_clean = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=False)
        spec_clean = inputs_clean.input_values.squeeze(0).numpy().T
        
        save_single_plot(
            spec_clean, 
            sampling_rate, 
            "AST - Clean (Mixed Mono, Auto Length)", 
            "ast_clean.png",
            x_limit=None
        )

        # --- 图2: AST Clean 10s (10s 视图) ---
        save_single_plot(
            spec_clean, 
            sampling_rate, 
            "AST - Clean (Mixed Mono, 10s View)", 
            "ast_clean_10s.png",
            x_limit=10.0
        )

        # --- 图3: AST Padded Full (完整模型输入) ---
        inputs_padded = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
        spec_padded = inputs_padded.input_values.squeeze(0).numpy().T

        save_single_plot(
            spec_padded, 
            sampling_rate, 
            "AST - Model Input (Padded, Full Length)", 
            "ast_padded_full.png",
            x_limit=None 
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error AST: {e}")

# 2. PaSST 模型处理 (双声道)
def generate_passt_images(audio_path):
    print("\n--- Processing PaSST ---")
    try:
        passt_model = get_basic_model(mode="logits")
        mel_transformer = passt_model.mel
        sampling_rate = 32000
        
        speech, sr = librosa.load(audio_path, sr=sampling_rate, mono=False)
        
        if speech.ndim == 1:
            speech = np.stack([speech, speech])
        elif speech.shape[0] > 2:
            speech = speech[:2, :]
        
        left_tensor = torch.from_numpy(speech.astype(np.float32))[0, :]
        right_tensor = torch.from_numpy(speech.astype(np.float32))[1, :]

        # --- 图4: PaSST Clean Stereo ---
        passt_model.eval() 
        spec_l_clean = mel_transformer(left_tensor.unsqueeze(0)).squeeze(0).detach().numpy()
        spec_r_clean = mel_transformer(right_tensor.unsqueeze(0)).squeeze(0).detach().numpy()
        
        save_dual_channel_plot(
            spec_l_clean, 
            spec_r_clean,
            sampling_rate,
            "PaSST Clean (Original Stereo)",
            "passt_clean_stereo.png",
            x_limit=None
        )
        
        # --- 图5: PaSST Padded Stereo (10s) ---
        passt_model.train() 
        target_len = 32000 * 10
        
        def pad_tensor(t, length):
            padded = torch.zeros(length)
            curr = min(len(t), length)
            padded[:curr] = t[:curr]
            return padded

        padded_l = pad_tensor(left_tensor, target_len)
        padded_r = pad_tensor(right_tensor, target_len)
        
        spec_l_padded = mel_transformer(padded_l.unsqueeze(0)).squeeze(0).detach().numpy()
        spec_r_padded = mel_transformer(padded_r.unsqueeze(0)).squeeze(0).detach().numpy()

        save_dual_channel_plot(
            spec_l_padded,
            spec_r_padded,
            sampling_rate,
            "PaSST Input (Padded & Masked)",
            "passt_padded_stereo.png",
            x_limit=10.0
        )

    except Exception as e:
        print(f"Error PaSST: {e}")

if __name__ == '__main__':
    if os.path.exists(AUDIO_FILE_PATH):
        generate_ast_images(AUDIO_FILE_PATH, AST_MODEL_PATH)
        generate_passt_images(AUDIO_FILE_PATH)
    else:
        print("Audio file not found.")
