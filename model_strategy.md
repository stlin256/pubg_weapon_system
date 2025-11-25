# 武器声音识别模型：策略与实施蓝图

本文档详细阐述了PUBG武器声音识别模块的技术选型、数据处理流程、模型构建细节以及训练与评估策略。

## 1. 最终模型选型与技术路径

为了兼顾课程设计要求、技术先进性和项目可行性，我们设定三层技术路径：

1.  **基线模型 (Baseline)**: **MFCC + 梯度提升树 (XGBoost)**
    *   **目的**: 满足课程Level A和B的基础要求，建立一个可靠的性能参照标准。
    *   **优势**: 技术成熟，实现快速，具备一定的可解释性（特征重要性分析）。

2.  **核心模型 (Primary Target)**: **微调音频声谱图Transformer (AST)**
    *   **目的**: 追求更高的识别准确率，探索深度学习在音频事件分类上的应用。
    *   **优势**: SOTA (State-of-the-Art) 架构，专为音频事件分类设计，能有效学习声谱图中的时频模式。我们将利用在大型音频数据集 (AudioSet) 上预训练过的模型进行微调，实现高效迁移学习。

3.  **探索模型 (Advanced Exploration)**: **微调HuBERT或Wav2Vec 2.0**
    *   **目的**: 挑战端到端的音频识别，尝试从原始波形直接学习。
    *   **优势**: 强大的自监督预训练模型，对音频有更底层的理解，在数据量有限时可能展现出更强的泛化能力。实现复杂度更高，作为项目的进阶探索和亮点。

**本项目将优先完成基线模型的构建，并将主要精力投入到核心模型（AST）的微调与优化上。**

---

## 2. 数据准备与预处理流程

无论是哪种模型，高质量、标准化的数据输入都是成功的关键。

```mermaid
graph TD
    A[原始音频 .mp3] --> B{标准化处理};
    B --> C[加载音频];
    B --> D[重采样至 16kHz];
    B --> E[音频增广 (可选)];

    subgraph "基线模型 Pipeline"
        E --> F[MFCC 特征提取];
        F --> G[特征向量];
    end

    subgraph "AST模型 Pipeline"
        E --> H[Mel 声谱图转换];
        H --> I[图像/张量];
    end

    subgraph "HuBERT模型 Pipeline"
        E --> J[原始波形];
        J --> K[一维张量];
    end

    G & I & K --> L{数据集划分};
    L --> M[训练集 Train];
    L --> N[验证集 Validation];
    L --> O[测试集 Test];
    M & N & O --> P[封装为 PyTorch DataLoader];
```

**关键步骤详解:**

1.  **加载与重采样**: 使用 `librosa` 库加载音频。所有音频将被重采样到 `16kHz`，这是大多数预训练音频模型的标准输入采样率，确保一致性。
2.  **音频增广 (Data Augmentation)**: 为防止模型过拟合，在训练时可以对音频进行实时增广，例如：
    *   随机增加背景噪音。
    *   随机改变音量 (Gain)。
    *   时间拉伸或压缩 (Time Stretch)。
3.  **特征提取**:
    *   **MFCC**: 使用 `librosa.feature.mfcc` 提取。通常提取 `20` 到 `40` 个系数，并计算其均值和标准差，形成一个固定长度的特征向量。
    *   **Mel 声谱图 (for AST)**: 使用 `librosa.feature.melspectrogram`。需要将输出的声谱图处理成与预训练AST模型输入要求一致的尺寸和格式。这将通过 `Hugging Face` 的 `ASTFeatureExtractor` 自动完成。
4.  **数据集划分**: 严格按照 `80:10:10` 或 `70:15:15` 的比例划分训练集、验证集和测试集，确保测试集的数据在训练和验证阶段**从未被模型见过**。
---
 
## 2.5 数据勘探结论与应对策略
 
基于对 `sounds/` 目录下训练集和测试集的分析，我们得出以下关键结论，并制定相应策略：
 
*   **核心挑战：严重的类别不均衡 (Class Imbalance)**
    *   **现象**: 数据集中 `ak`, `m4`, `m24` 等武器的样本量远超其他类别，部分武器样本甚至只有个位数。
    *   **风险**: 直接训练会导致模型严重偏向多数类，在稀有类别上性能极差，最终得到虚高的、不可靠的评估指标。
 
*   **综合应对策略**:
    1.  **数据增广 (Data Augmentation)**: 这是我们的主要手段。在加载数据时，对**所有样本**（特别是少数类样本）进行实时的、随机的音频变换。例如使用 `audiomentations` 库实现：
        *   `AddGaussianNoise`: 增加高斯噪声。
        *   `TimeStretch`: 时间拉伸。
        *   `PitchShift`: 音高变换。
        *   `Gain`: 调整音量。
    2.  **加权采样 (Weighted Sampling)**: 在构建 `DataLoader` 时，使用 `WeightedRandomSampler`。为样本量少的类别赋予更高的采样权重，确保在一个训练批次(batch)中，模型见到各类样本的概率大致相等。
    3.  **加权损失函数 (Weighted Loss Function)**: 在计算损失时，为不同类别分配权重。可以简单地使用与类别样本数成反比的权重，在 `CrossEntropyLoss` 中通过 `weight` 参数传入。
 
*   **评估策略修正**:
    *   **首要指标**: 我们将**宏平均F1分数 (F1-Macro)** 和 **宏平均召回率 (Recall-Macro)** 作为比**总体准确率 (Accuracy)** 更重要的核心评估指标。它们能更公平地反映模型在所有类别（包括稀有类别）上的综合性能。
    *   **必要分析**: 必须生成并仔细分析**混淆矩阵**，以直观地看出模型具体在哪些类别之间产生了混淆。
 
---
 
## 3. 模型实现、训练与微调
 

我们将主要使用 `PyTorch` 框架和 `Hugging Face Transformers` 库。

### 3.1 基线模型 (XGBoost)

*   **实现**: 使用 `scikit-learn` 和 `xgboost` 库。
*   **流程**:
    1.  对所有音频文件预先提取MFCC特征，并保存为`.csv`或`.npy`文件。
    2.  加载特征和标签。
    3.  训练 `XGBClassifier` 模型。
    4.  使用 `GridSearchCV` 进行超参数搜索以优化模型。

### 3.2 核心模型 (AST 微调)

这是项目的重点。我们将利用 `Hugging Face` 提供的生态。

*   **模型选择**: `MIT/ast-finetuned-audioset-10-10-0.4593` 或其他在 AudioSet 上微调过的版本。
*   **微调流程**:
    1.  **加载处理器**: `ASTFeatureExtractor.from_pretrained(...)`，它会负责将原始音频处理成模型所需的声谱图格式。
    2.  **加载模型**: `ASTForAudioClassification.from_pretrained(...)`。
    3.  **修改分类头**: 预训练模型的分类头（classifier layer）是为AudioSet的527个类别设计的。我们必须将其替换为一个新的线性层，其输出维度等于我们武器类别的数量（例如10种武器，输出维度就是10）。
        ```python
        # 伪代码示例
        from transformers import ASTForAudioClassification
        
        num_labels = len(weapon_classes)
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=num_labels, ignore_mismatched_sizes=True)
        ```
    4.  **使用 `Trainer` API 进行训练**: 这是 `Hugging Face` 提供的强大工具，能极大简化训练流程。
        *   定义 `TrainingArguments`: 设置输出目录、学习率、批次大小、训练轮数、评估策略等。
        *   定义评估函数: 计算 `accuracy`, `f1_macro` 等指标。
        *   实例化 `Trainer` 对象，传入模型、参数、数据集和评估函数。
        *   调用 `trainer.train()` 开始训练，`trainer.evaluate()` 进行评估。

### 3.3 训练策略

*   **优化器**: `AdamW`，这是 Transformer 模型的标准选择。
*   **学习率调度器**: 使用线性预热（warm-up）和衰减的策略，有助于模型稳定收敛。
*   **损失函数**: 交叉熵损失 (`CrossEntropyLoss`)。
*   **实验跟踪**: 强烈建议使用 `TensorBoard` 或 `Weights & Biases` 记录训练过程中的损失和评估指标变化，便于分析和调优。

---

## 4. 评估与部署

*   **评估指标**: 严格遵守课程要求，使用 `准确率(accuracy)`、`宏平均精确率(precision_macro)`、`宏平均召回率(recall_macro)` 和 `宏平均F1分数(f1_macro)`。同时，绘制并分析**混淆矩阵**，以了解模型具体在哪些类别上容易犯错。
*   **模型保存**: 保存训练好的模型权重（特别是效果最好的checkpoint）和配置文件，以便后续在Web应用中加载和使用。
*   **推理 (Inference)**: 在 `FastAPI` 后端，我们将创建一个推理函数：
    1.  接收上传的音频文件。
    2.  加载我们训练好的模型和处理器。
    3.  对音频进行预处理。
    4.  将处理后的数据送入模型进行预测。
    5.  返回预测结果（武器名称和置信度）。