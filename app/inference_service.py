import os
import glob
import joblib
import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from hear21passt.base import get_basic_model

# --- 项目特定的配置 ---
# 假设此服务在 Flask 应用上下文中运行，可以访问项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'trained_models')
MAX_PADDING_LENGTH = 174 # From feature_extractor.py
SAMPLING_RATE = 16000 # From config.py

import json

class InferenceService:
    def __init__(self):
        self._model_cache = {}
        self.strategy_file = 'cache_strategy.json'
        self.cache_strategy = self._load_strategy()
        self._available_models = self._scan_models()
        self._label_encoders = self._load_label_encoders()
        if self.cache_strategy == 'all':
            self.preload_all_models()

    def _load_strategy(self):
        if os.path.exists(self.strategy_file):
            with open(self.strategy_file, 'r') as f:
                return json.load(f).get('strategy', 'all')
        return 'all'

    def _save_strategy(self):
        with open(self.strategy_file, 'w') as f:
            json.dump({'strategy': self.cache_strategy}, f)

    def set_cache_strategy(self, strategy):
        if strategy in ['all', 'selected']:
            self.cache_strategy = strategy
            self._save_strategy()
            if strategy == 'all':
                self.preload_all_models()
            else:
                self._model_cache.clear()
        return self.cache_strategy

    def preload_model(self, model_name):
        """主动加载单个模型到缓存。"""
        if model_name not in self._model_cache:
            self._load_model(model_name)
    
    def preload_all_models(self):
        """预加载所有发现的模型。"""
        for target in self._available_models:
            for model_name in self._available_models[target]:
                self.preload_model(model_name)

    def _scan_models(self):
        """扫描模型目录，按任务对模型进行分类。"""
        models = {"weapon": [], "distance": [], "direction": []}
        
        # 扫描 scikit-learn 模型 (.pkl)
        pkl_files = glob.glob(os.path.join(MODEL_SAVE_PATH, '*.pkl'))
        for f in pkl_files:
            if 'label_encoder' in f: continue
            model_name = os.path.basename(f).replace('.pkl', '')
            parts = model_name.split('_')
            if len(parts) > 1:
                target = parts[1]
                if target in models:
                    models[target].append(model_name)

        # 扫描 Transformer 模型 (目录)
        model_dirs = [d for d in os.listdir(MODEL_SAVE_PATH) if os.path.isdir(os.path.join(MODEL_SAVE_PATH, d))]
        for d in model_dirs:
            if 'finetuned' in d or 'checkpoints' in d: continue
            parts = d.split('_')
            target = parts[1] if 'passt' in d else parts[2 if 'stereo' in d else 1]
            if target in models:
                models[target].append(d)

        return models

    def _load_label_encoders(self):
        """加载所有可用的 LabelEncoder。"""
        encoders = {}
        for target in ["weapon", "distance", "direction"]:
            path = os.path.join(MODEL_SAVE_PATH, f'label_encoder_{target}.pkl')
            if os.path.exists(path):
                encoders[target] = joblib.load(path)
        return encoders

    def get_available_models(self):
        return self._available_models

    def _load_model(self, model_name):
        """根据模型名称加载模型，并实现缓存。"""
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = None

        # .pkl 模型文件名不含扩展名，所以我们直接拼接
        pkl_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.pkl")
        if os.path.exists(pkl_path):
            model = joblib.load(pkl_path)
        
        elif 'ast' in model_name:
            model_path = os.path.join(MODEL_SAVE_PATH, model_name)
            model = AutoModelForAudioClassification.from_pretrained(model_path).to(device)
            model.eval()

        elif 'passt' in model_name:
            model_path = os.path.join(MODEL_SAVE_PATH, model_name)
            with open(os.path.join(model_path, "classes.txt"), "r") as f:
                num_labels = len([line.strip() for line in f.readlines()])

            passt_model = get_basic_model(mode="logits")
            old_conv = passt_model.net.patch_embed.proj
            new_conv = nn.Conv2d(in_channels=2, out_channels=old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)
            passt_model.net.patch_embed.proj = new_conv
            old_head = passt_model.net.head[1]
            passt_model.net.head[1] = nn.Linear(old_head.in_features, num_labels)
            
            state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
            passt_model.net.load_state_dict(torch.load(state_dict_path, map_location=device))
            passt_model.to(device)
            passt_model.eval()
            model = passt_model
            
        if model and self.cache_strategy == 'all':
            self._model_cache[model_name] = model
        return model

    def predict(self, audio_path, target, model_name):
        model = self._load_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} could not be loaded.")
        
        encoder = self._label_encoders.get(target)
        if not encoder:
            raise ValueError(f"LabelEncoder for target {target} not found.")

        # --- 特征提取和推理 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 统一逻辑：通过是否存在 .pkl 文件来判断模型类型
        pkl_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.pkl")
        if os.path.exists(pkl_path):
            audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            if mfccs.shape[1] > MAX_PADDING_LENGTH: mfccs = mfccs[:, :MAX_PADDING_LENGTH]
            else: mfccs = np.pad(mfccs, ((0, 0), (0, MAX_PADDING_LENGTH - mfccs.shape[1])), mode='constant')
            features = mfccs.reshape(1, -1)
            prediction_idx = model.predict(features)[0]

        elif 'ast' in model_name:
            feature_extractor = AutoFeatureExtractor.from_pretrained(os.path.join(MODEL_SAVE_PATH, model_name))
            speech, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
            inputs = feature_extractor(speech, sampling_rate=SAMPLING_RATE, padding="max_length", return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            prediction_idx = torch.argmax(logits, dim=-1).item()

        elif 'passt' in model_name:
            speech, sr = librosa.load(audio_path, sr=32000, mono=False)
            if speech.ndim == 1: speech = np.stack([speech, speech])
            
            inputs = torch.from_numpy(speech.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                x_left, x_right = inputs[:, 0], inputs[:, 1]
                spec_left = model.mel(x_left).unsqueeze(1)
                spec_right = model.mel(x_right).unsqueeze(1)
                x = torch.cat([spec_left, spec_right], dim=1)
                logits = model.net(x)
                if isinstance(logits, tuple): logits = logits[0]
            prediction_idx = torch.argmax(logits, dim=-1).item()
        
        return encoder.inverse_transform([prediction_idx])[0]

# 创建一个单例
inference_service = InferenceService()