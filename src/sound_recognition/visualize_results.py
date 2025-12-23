import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.font_manager import FontManager
from . import config

# --- 全局配置 ---
EVALUATION_CSV_PATH = os.path.join(config.REPORTS_PATH, 'evaluation_results.csv')
FIGURES_SAVE_PATH = config.FIGURES_PATH
HIGH_DPI = 300

# --- 1. 设置全局绘图样式 ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- 2. 动态字体配置 ---
def set_chinese_font():
    """在系统中查找可用的中文字体"""
    fm = FontManager()
    font_preferences = [
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
        'SimHei', 'Microsoft YaHei', 'PingFang SC'
    ]
    
    found_font = None
    available_fonts = set(f.name for f in fm.ttflist)
    
    for font_name in font_preferences:
        if font_name in available_fonts:
            print(f"✅ 成功找到中文字体: {font_name}")
            found_font = font_name
            break
            
    if found_font:
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [found_font] + plt.rcParams['font.sans-serif'],
            'axes.unicode_minus': False
        })
        return found_font
    else:
        print("⚠️ 警告: 未找到推荐的中文字体。")
        return None

CHINESE_FONT = set_chinese_font()

# --- 3. 翻译字典 ---
TRANSLATIONS = {
    'en': {
        'model': 'Model', 'target': 'Target', 'accuracy': 'Accuracy',
        'precision (macro)': 'Precision (Macro)', 'recall (macro)': 'Recall (Macro)',
        'f1-score (macro)': 'F1-Score (Macro)',
        'all_title': 'Model Comparison: {target}',
        'ml_title': 'Traditional ML Models: {target}',
        'transformer_title': 'Transformer Models: {target}',
        'grouped_accuracy_title': 'Overall Accuracy Overview (All Targets)',
        'grouped_f1_title': 'Overall F1-Score (Macro) Overview (All Targets)'
    },
    'zh': {
        'model': '模型', 
        'target': 'Target', 
        'accuracy': '准确率',
        'precision (macro)': '精确率 (Macro)',
        'recall (macro)': '召回率 (Macro)',
        'f1-score (macro)': 'F1分数 (Macro)',
        'all_title': '目标: "{target}" - 全模型性能对比',
        'ml_title': '目标: "{target}" - 传统机器学习模型对比',
        'transformer_title': '目标: "{target}" - Transformer 模型对比',
        'grouped_accuracy_title': '所有目标类别的模型准确率总览',
        'grouped_f1_title': '所有目标类别的模型F1分数(Macro)总览'
    }
}

def classify_model_type(model_name):
    if 'ast' in model_name or 'passt' in model_name:
        return 'Transformer'
    return 'Traditional ML'

def plot_comparison(df, title_key, filename, lang='en', target_en=None, x='model', y='accuracy', hue=None):
    """单独目标的详细对比图"""
    t = TRANSLATIONS[lang]
    fig, ax = plt.subplots(figsize=(18, 10))
    palette = sns.color_palette("viridis", n_colors=len(df))
    
    if hue is None:
        barplot = sns.barplot(data=df, x=x, y=y, hue=x, palette=palette, width=0.8, legend=False, ax=ax)
    else:
        barplot = sns.barplot(data=df, x=x, y=y, hue=hue, palette='viridis', width=0.8, ax=ax)
    
    for p in barplot.patches:
        height = p.get_height()
        if not np.isnan(height):
            barplot.annotate(format(height, '.3f'), 
                           (p.get_x() + p.get_width() / 2., height), 
                           ha='center', va='center', xytext=(0, 9), 
                           textcoords='offset points', fontsize=11, weight='bold')

    display_target = target_en.capitalize() if target_en else ""
    title_text = t[title_key].format(target=display_target)
    ax.set_title(title_text, fontsize=24, weight='bold', pad=25)
    ax.set_ylabel(t[y], fontsize=16)
    ax.set_xlabel(t[x], fontsize=16)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=13)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    save_path = os.path.join(FIGURES_SAVE_PATH, filename)
    plt.savefig(save_path, dpi=HIGH_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {save_path}")

def plot_grouped_overall(df, title_key, filename, lang='en', y='accuracy'):
    """
    全模型总览图 
    """
    t = TRANSLATIONS[lang]
    df_plot = df.copy()

    df_plot['target'] = df_plot['target'].str.capitalize()
    target_order = ['Direction', 'Distance', 'Weapon']
    existing_targets = [tg for tg in target_order if tg in df_plot['target'].unique()]
    
    df_plot['x_label'] = df_plot['model'] + '_' + df_plot['target'].str.lower()
    
    df_plot['target_cat'] = pd.Categorical(df_plot['target'], categories=existing_targets, ordered=True)
    df_sorted = df_plot.sort_values(by=['target_cat', y], ascending=[True, False])

    plt.figure(figsize=(22, 8))
    
    bar_plot = sns.barplot(
        data=df_sorted,
        x='x_label', y=y, hue='target', 
        palette='viridis', dodge=False, width=0.8
    )

    plt.title(t[title_key], fontsize=24, weight='bold', pad=20)
    plt.xlabel(t['model'], fontsize=16, labelpad=10)
    plt.ylabel(t[y], fontsize=16, labelpad=10)
    
    # Y轴 0-1
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=13)
    plt.ylim(0, 1.08)
    
    plt.xticks(rotation=45, ha='right', fontsize=11)

    for p in bar_plot.patches:
        height = p.get_height()
        if height > 0:
            bar_plot.annotate(format(height, '.3f'), 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', 
                        xytext=(0, 6), 
                        textcoords='offset points',
                        fontsize=9)

    plt.legend(
        title=t['target'], 
        title_fontsize=12, 
        fontsize=11, 
        loc='upper right',
        frameon=True,         # <--- 必须开启边框，否则背景色不显示
        facecolor='#E0E0E0',  # 浅灰色背景
        edgecolor='None',     # 无边框线条
        framealpha=1          # 不透明
    )

    save_path = os.path.join(FIGURES_SAVE_PATH, filename)
    plt.savefig(save_path, dpi=HIGH_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved overall grouped plot to: {save_path}")

def plot_dual_metric_comparison(df, title, filename, lang='en'):
    """
    生成双指标对比图 (条形图 + 折线图).
    """
    t = TRANSLATIONS[lang]
    df_sorted = df.sort_values(by='accuracy', ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(18, 10))

    # --- 优化 1：使用更有梯度的调色板 ---
    palette = sns.color_palette("viridis", n_colors=len(df_sorted))
    # 绘制准确率条形图 (禁用自动图例)
    bar = sns.barplot(data=df_sorted, x='model', y='accuracy', hue='model', palette=palette, ax=ax1, label=t['accuracy'], legend=False)
    ax1.set_xlabel(t['model'], fontsize=16)
    ax1.set_ylabel(t['accuracy'], fontsize=16, color='cornflowerblue')
    ax1.tick_params(axis='y', labelcolor='cornflowerblue')
    
    # 修复：'ha' 不是 tick_params 的有效参数，应使用 setp
    ax1.tick_params(axis='x', labelsize=13)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)

    # 在条形图上标注数值
    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points',
                     fontsize=10, weight='bold', color='black')

    # 创建第二个Y轴绘制F1分数折线图
    ax2 = ax1.twinx()
    line = sns.lineplot(data=df_sorted, x='model', y='f1-score (macro)', marker='o', sort=False,
                 ax=ax2, color='darkorange', label=t['f1-score (macro)'])
    ax2.set_ylabel(t['f1-score (macro)'], fontsize=16, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1.1)
    
    plt.title(title, fontsize=24, weight='bold', pad=25)
    
    # --- 优化 2：将图例置于内部右上角，并添加灰色背景 ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if ax1.legend_: ax1.legend_.remove()
    if ax2.legend_: ax2.legend_.remove()

    # 创建一个统一的图例
    ax2.legend(handles=handles1 + handles2, labels=[t['accuracy'], t['f1-score (macro)']],
               loc='upper right', frameon=True, facecolor='#E0E0E0', framealpha=0.8)

    fig.tight_layout() # 使用 tight_layout 自动调整边距

    save_path = os.path.join(FIGURES_SAVE_PATH, filename)
    plt.savefig(save_path, dpi=HIGH_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved dual metric plot to: {save_path}")

def plot_performance_heatmap(df, title, filename, lang='en'):
    """
    生成模型在不同任务上的性能热力图.
    """
    t = TRANSLATIONS[lang]
    df_copy = df.copy()
    
    # --- 使用正则表达式提取基础模型名称 ---
    df_copy['base_model'] = df_copy['model'].str.extract(r'(^[^_]+)')[0]
    
    # 创建数据透视表
    heatmap_data = df_copy.pivot_table(index='base_model', columns='target', values='f1-score (macro)')
    heatmap_data = heatmap_data.sort_values(by='weapon', ascending=False) # 按主要任务排序
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
    
    plt.title(title, fontsize=20, weight='bold', pad=20)
    plt.xlabel(t['target'], fontsize=14)
    plt.ylabel(t['model'], fontsize=14)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    save_path = os.path.join(FIGURES_SAVE_PATH, filename)
    plt.savefig(save_path, dpi=HIGH_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved performance heatmap to: {save_path}")

def main():
    print("--- Starting Results Visualization ---")
    if not os.path.exists(EVALUATION_CSV_PATH):
        print(f"Error: Evaluation file not found at {EVALUATION_CSV_PATH}")
        return

    df = pd.read_csv(EVALUATION_CSV_PATH)
    df['model'] = df['model'].str.replace('_GridSearch', '_GS')
    df['model_type'] = df['model'].apply(classify_model_type)

    for target in df['target'].unique():
        df_target = df[df['target'] == target].sort_values('accuracy', ascending=False)
        df_ml = df_target[df_target['model_type'] == 'Traditional ML']
        df_transformer = df_target[df_target['model_type'] == 'Transformer']

        # 英文单图
        plot_comparison(df_target, 'all_title', f'all_models_{target}.png', lang='en', target_en=target)
        plot_comparison(df_ml, 'ml_title', f'ml_models_{target}.png', lang='en', target_en=target)
        plot_comparison(df_transformer, 'transformer_title', f'transformer_models_{target}.png', lang='en', target_en=target)

        # 中文单图
        plot_comparison(df_target, 'all_title', f'all_models_{target}_zh.png', lang='zh', target_en=target)
        plot_comparison(df_ml, 'ml_title', f'ml_models_{target}_zh.png', lang='zh', target_en=target)
        plot_comparison(df_transformer, 'transformer_title', f'transformer_models_{target}_zh.png', lang='zh', target_en=target)
        
    # 总览图
    plot_grouped_overall(df, 'grouped_accuracy_title', 'overall_accuracy_grouped.png', lang='en')
    plot_grouped_overall(df, 'grouped_f1_title', 'overall_f1_score_grouped.png', lang='en', y='f1-score (macro)')
    
    # 中文总览图
    plot_grouped_overall(df, 'grouped_accuracy_title', 'overall_accuracy_grouped_zh.png', lang='zh')
    plot_grouped_overall(df, 'grouped_f1_title', 'overall_f1_score_grouped_zh.png', lang='zh', y='f1-score (macro)')
    
    # --- 1. 生成新的双指标对比图 ---
    for target in df['target'].unique():
        df_target = df[df['target'] == target]
        # 英文版
        plot_dual_metric_comparison(
            df_target,
            f'Dual Metric Comparison: {target.capitalize()}',
            f'dual_metric_{target}.png',
            lang='en'
        )
        # 中文版
        plot_dual_metric_comparison(
            df_target,
            f'双指标对比: {target.capitalize()}',
            f'dual_metric_{target}_zh.png',
            lang='zh'
        )

    # --- 2. 生成新的性能热力图 ---
    # 英文版
    plot_performance_heatmap(
        df,
        'Model Performance Heatmap (F1-Score)',
        'performance_heatmap.png',
        lang='en'
    )
    # 中文版
    plot_performance_heatmap(
        df,
        '模型性能热力图 (F1分数)',
        'performance_heatmap_zh.png',
        lang='zh'
    )

    print("\n--- Visualization complete. Figures saved to 'reports/figures/'. ---")

if __name__ == '__main__':
    main()