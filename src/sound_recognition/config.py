import os

# --- 路径定义 ---

# 项目根目录
# 使用 os.path.abspath 和 __file__ 来确保无论从哪里运行脚本，路径都是正确的
# __file__ -> 当前文件路径 (config.py)
# os.path.dirname -> 获取文件所在目录 (sound_recognition)
# os.path.join(..., '..', '..') -> 上溯两级到项目根目录
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

# 原始数据路径
# 假设您的原始音频文件存放在项目根目录下的 'sounds/' 文件夹中
AUDIO_DATA_PATH = os.path.join(BASE_DIR, 'sounds')

# 特征保存路径
# 我们将把提取的 MFCC 特征保存在一个新创建的 'features' 目录下
FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'features')
os.makedirs(FEATURES_PATH, exist_ok=True) # 确保目录存在

# 训练好的模型的保存路径
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'trained_models')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True) # 确保目录存在

# 评估报告的保存路径
REPORTS_PATH = os.path.join(BASE_DIR, 'reports')
FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures')
os.makedirs(FIGURES_PATH, exist_ok=True) # 确保目录存在

# --- 音频处理超参数 ---
SAMPLING_RATE = 16000   # 采样率
N_MFCC = 13             # MFCC系数的数量
HOP_LENGTH = 512        # 帧移
N_FFT = 2048            # FFT窗口大小

# --- 数据集划分参数 ---
TEST_SIZE = 0.2         # 测试集比例
VALIDATION_SIZE = 0.1   # 验证集比例 (在训练集中划分)
RANDOM_STATE = 42       # 随机种子，确保可复现性

print(f"Project Base Directory: {BASE_DIR}")
print(f"Audio Data Path: {AUDIO_DATA_PATH}")