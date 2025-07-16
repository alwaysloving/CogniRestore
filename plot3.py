import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import os

# 设置学术风格的可视化参数
try:
    # 尝试使用seaborn样式，如果可用的话
    sns.set_theme(style="whitegrid")
except:
    # 如果seaborn不可用或版本不兼容，使用matplotlib的基本设置
    plt.style.use('default')
    plt.grid(True, linestyle='--', alpha=0.7)

# 设置字体和其他参数
try:
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
except:
    # 如果字体设置失败，使用默认字体
    pass

# 设置线条宽度等参数
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2

# 颜色方案 - 使用学术刊物常用的配色
COLORS = {
    'loss': '#B22222',         # 深红色
    'accuracy': '#1f77b4',     # 蓝色
    'depth': '#2ca02c',        # 绿色
    'image': '#ff7f0e',        # 橙色
    'text': '#9467bd',         # 紫色
    'top50': '#8c564b',        # 棕色
    'top100': '#e377c2',       # 粉色
    'top200': '#7f7f7f',       # 灰色
    'top5': '#17becf',         # 青色
    'grid': '#cccccc',         # 网格线颜色
    'background': '#f9f9f9'    # 背景色
}

def load_data(file_path="/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/csv/version_0/metrics.csv"):
    """
    从CSV文件中加载训练数据
    
    参数:
        file_path: CSV文件路径
        
    返回:
        pandas DataFrame: 包含训练指标的数据框
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到CSV文件: {file_path}")
            
        # 读取CSV文件
        data = pd.read_csv(file_path)
        
        # 基本验证
        required_columns = ['epoch', 'loss_sum']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要的列: {', '.join(missing_columns)}")
            
        print(f"成功从 {file_path} 加载了 {len(data)} 条训练记录")
        return data
        
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

def create_training_progress_figure(data):
    """创建训练进度综合图表"""
    # 过滤只包含完整训练记录的行
    filtered_data = data[~data['epoch'].duplicated(keep='first')]
    train_data = data[data['step'].duplicated(keep='last')]
    
    # 创建图形和子图布局
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # 顶部大图: 损失和总体准确率
    ax_main = fig.add_subplot(gs[0, :])
    
    # 左下: 模态准确率比较
    ax_modality = fig.add_subplot(gs[1, 0])
    
    # 右下: Top-K准确率
    ax_topk = fig.add_subplot(gs[1, 1])
    
    # 底部: 训练/验证损失
    ax_trainval = fig.add_subplot(gs[2, :])
    
    # 设置图表背景
    for ax in [ax_main, ax_modality, ax_topk, ax_trainval]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    # === 主图: 损失和总体准确率 ===
    # 损失曲线
    loss = gaussian_filter1d(filtered_data['loss_sum'], sigma=1.0)
    ax_main.plot(filtered_data['epoch'], loss, color=COLORS['loss'], linewidth=2.5, 
                 label='Total Loss')
    
    ax_main.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Loss', fontsize=12, fontweight='bold', color=COLORS['loss'])
    ax_main.tick_params(axis='y', labelcolor=COLORS['loss'])
    
    # 右侧轴: 准确率
    ax_main2 = ax_main.twinx()
    
    # Top-100 整体准确率
    accuracy = gaussian_filter1d(filtered_data['top100class_accuracy/all'], sigma=1.0)
    ax_main2.plot(filtered_data['epoch'], accuracy, color=COLORS['accuracy'], linewidth=2.5, 
                  label='Overall Accuracy (top-100)')
    
    # 设置右侧Y轴
    ax_main2.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color=COLORS['accuracy'])
    ax_main2.tick_params(axis='y', labelcolor=COLORS['accuracy'])
    ax_main2.set_ylim(0, 1.05)
    ax_main2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    
    # 图例
    lines1, labels1 = ax_main.get_legend_handles_labels()
    lines2, labels2 = ax_main2.get_legend_handles_labels()
    ax_main.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    
    ax_main.set_title('Multi-Modal BCI Training Progress', 
                      fontsize=14, fontweight='bold')
    
    # === 左下图: 模态准确率 ===
    # 绘制各模态的准确率
    modality_columns = ['top100class_accuracy/depth', 'top100class_accuracy/image', 'top100class_accuracy/text']
    modality_labels = ['Depth', 'Image', 'Text']
    modality_colors = [COLORS['depth'], COLORS['image'], COLORS['text']]
    
    for i, col in enumerate(modality_columns):
        values = gaussian_filter1d(filtered_data[col], sigma=1.0)
        ax_modality.plot(filtered_data['epoch'], values, color=modality_colors[i], linewidth=2, 
                         label=f'{modality_labels[i]} Accuracy')
    
    ax_modality.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax_modality.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
    ax_modality.set_ylim(0, 1.05)
    ax_modality.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_modality.set_title('Accuracy by Modality (top-100)', 
                          fontsize=12, fontweight='bold')
    ax_modality.legend(loc='lower right', fontsize=9)
    
    # === 右下图: Top-K准确率 ===
    # 绘制不同top-k的整体准确率
    topk_columns = ['top50class_accuracy/all', 'top100class_accuracy/all', 'top200class_accuracy/all']
    topk_labels = ['top-50', 'top-100', 'top-200']
    topk_colors = [COLORS['top50'], COLORS['top100'], COLORS['top200']]
    
    for i, col in enumerate(topk_columns):
        values = gaussian_filter1d(filtered_data[col], sigma=1.0)
        ax_topk.plot(filtered_data['epoch'], values, color=topk_colors[i], linewidth=2, 
                     label=f'{topk_labels[i]} Accuracy')
    
    ax_topk.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax_topk.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
    ax_topk.set_ylim(0, 1.05)
    ax_topk.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_topk.set_title('Accuracy by Classification Granularity', 
                      fontsize=12, fontweight='bold')
    ax_topk.legend(loc='lower right', fontsize=9)
    
    # === 底部图: 训练/验证损失 ===
    # 绘制深度模态
    if 'train/loss_depth' in train_data.columns and 'val/loss_depth' in filtered_data.columns:
        ax_trainval.plot(train_data['epoch'], train_data['train/loss_depth'], color=COLORS['depth'], 
                         linestyle='--', linewidth=1.5, label='Depth Train Loss')
        ax_trainval.plot(filtered_data['epoch'], filtered_data['val/loss_depth'], color=COLORS['depth'], 
                         linewidth=2, label='Depth Val Loss')
    
    # 绘制图像模态
    if 'train/loss_img' in train_data.columns and 'val/loss_img' in filtered_data.columns:
        ax_trainval.plot(train_data['epoch'], train_data['train/loss_img'], color=COLORS['image'], 
                         linestyle='--', linewidth=1.5, label='Image Train Loss')
        ax_trainval.plot(filtered_data['epoch'], filtered_data['val/loss_img'], color=COLORS['image'], 
                         linewidth=2, label='Image Val Loss')
    
    # 绘制文本模态
    if 'train/loss_text' in train_data.columns and 'val/loss_text' in filtered_data.columns:
        ax_trainval.plot(train_data['epoch'], train_data['train/loss_text'], color=COLORS['text'], 
                         linestyle='--', linewidth=1.5, label='Text Train Loss')
        ax_trainval.plot(filtered_data['epoch'], filtered_data['val/loss_text'], color=COLORS['text'], 
                         linewidth=2, label='Text Val Loss')
    
    ax_trainval.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax_trainval.set_ylabel('Loss', fontsize=10, fontweight='bold')
    ax_trainval.set_title('Training vs. Validation Loss by Modality', 
                          fontsize=12, fontweight='bold')
    ax_trainval.legend(loc='upper right', fontsize=9)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def create_performance_summary_figure(data):
    """创建性能总结图表"""
    # 过滤只包含完整训练记录的行
    filtered_data = data[~data['epoch'].duplicated(keep='first')]
    
    # 获取最后一个epoch的记录
    final_metrics = filtered_data.iloc[-1]
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 顶部：最终性能横条图
    ax_bar = fig.add_subplot(gs[0, :])
    
    # 左下：Top-5准确率对比
    ax_top5 = fig.add_subplot(gs[1, 0])
    
    # 右下：最终数值摘要
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis('off')  # 隐藏坐标轴
    
    # 设置图表背景
    for ax in [ax_bar, ax_top5]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    # === 顶部：最终性能横条图 ===
    metrics = [
        'top50class_accuracy/all', 'top100class_accuracy/all', 'top200class_accuracy/all',
        'top50class_accuracy/depth', 'top100class_accuracy/depth', 'top200class_accuracy/depth',
        'top50class_accuracy/image', 'top100class_accuracy/image', 'top200class_accuracy/image',
        'top50class_accuracy/text', 'top100class_accuracy/text', 'top200class_accuracy/text'
    ]
    
    metric_labels = [
        'All (top-50)', 'All (top-100)', 'All (top-200)',
        'Depth (top-50)', 'Depth (top-100)', 'Depth (top-200)',
        'Image (top-50)', 'Image (top-100)', 'Image (top-200)',
        'Text (top-50)', 'Text (top-100)', 'Text (top-200)'
    ]
    
    # 提取值
    values = [final_metrics[m] for m in metrics]
    
    # 设置颜色方案
    modality_colors = plt.cm.viridis(np.linspace(0, 0.9, 4))  # 4种颜色对应all, depth, image, text
    colors = []
    for i in range(4):
        for _ in range(3):
            colors.append(modality_colors[i])
    
    # 画横条图
    bars = ax_bar.barh(metric_labels, values, color=colors, height=0.6, alpha=0.8)
    
    # 添加标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax_bar.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1%}', va='center', fontsize=10, fontweight='bold')
    
    ax_bar.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax_bar.set_xlim(0, 1.05)
    ax_bar.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_bar.set_title(f'Final Performance Metrics (Epoch {int(final_metrics["epoch"])})', 
                     fontsize=14, fontweight='bold')
    
    # === 左下图：Top-5准确率对比 ===
    # 绘制各模态的Top-5准确率曲线
    top5_columns = ['top100class_top5_acc/depth', 'top100class_top5_acc/image', 'top100class_top5_acc/text']
    top5_labels = ['Depth Top-5', 'Image Top-5', 'Text Top-5']
    top5_colors = [COLORS['depth'], COLORS['image'], COLORS['text']]
    
    for i, col in enumerate(top5_columns):
        values = gaussian_filter1d(filtered_data[col], sigma=1.0)
        ax_top5.plot(filtered_data['epoch'], values, color=top5_colors[i], linewidth=2, 
                     label=f'{top5_labels[i]}')
    
    ax_top5.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax_top5.set_ylabel('Top-5 Accuracy', fontsize=10, fontweight='bold')
    ax_top5.set_ylim(0, 1.05)
    ax_top5.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_top5.set_title('Top-5 Accuracy by Modality (top-100)', 
                      fontsize=12, fontweight='bold')
    ax_top5.legend(loc='lower right', fontsize=9)
    
    # === 右下图：结果摘要文本 ===
    # 计算关键指标的变化
    initial_epoch = filtered_data.iloc[0]
    loss_change = ((initial_epoch['loss_sum'] - final_metrics['loss_sum']) 
                  / initial_epoch['loss_sum'] * 100)
    acc_change = (final_metrics['top100class_accuracy/all'] - initial_epoch['top100class_accuracy/all']) * 100
    
    # 创建结果摘要文本
    summary_text = (
        f"TRAINING SUMMARY (Epochs: {len(filtered_data)})\n\n"
        f"LOSS REDUCTION:\n"
        f"  Initial: {initial_epoch['loss_sum']:.4f}\n"
        f"  Final: {final_metrics['loss_sum']:.4f}\n"
        f"  Change: -{loss_change:.1f}%\n\n"
        f"ACCURACY IMPROVEMENT (Top-100):\n"
        f"  Initial: {initial_epoch['top100class_accuracy/all']*100:.1f}%\n"
        f"  Final: {final_metrics['top100class_accuracy/all']*100:.1f}%\n"
        f"  Change: +{acc_change:.1f} points\n\n"
        f"MODALITY PERFORMANCE (Final):\n"
        f"  Depth: {final_metrics['top100class_accuracy/depth']*100:.1f}%\n"
        f"  Image: {final_metrics['top100class_accuracy/image']*100:.1f}%\n"
        f"  Text: {final_metrics['top100class_accuracy/text']*100:.1f}%\n\n"
        f"TOP-5 PERFORMANCE (Final):\n"
        f"  Depth: {final_metrics['top100class_top5_acc/depth']*100:.1f}%\n"
        f"  Image: {final_metrics['top100class_top5_acc/image']*100:.1f}%\n"
        f"  Text: {final_metrics['top100class_top5_acc/text']*100:.1f}%"
    )
    
    # 添加摘要文本
    ax_text.text(0.05, 0.95, summary_text, va='top', fontsize=12, 
                 backgroundcolor=COLORS['background'], 
                 bbox=dict(facecolor=COLORS['background'], edgecolor='#cccccc', 
                           boxstyle='round,pad=1', alpha=0.9))
    
    fig.suptitle("Multi-Modal BCI Model Performance Summary", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

def save_results_summary(data, output_file='bci_results_summary.txt'):
    """
    保存训练结果的摘要统计信息到文本文件
    
    参数:
        data: 训练数据的DataFrame
        output_file: 输出文件的路径
    """
    # 获取过滤后的数据
    filtered_data = data[~data['epoch'].duplicated(keep='first')]
    
    # 获取初始和最终epoch的数据
    initial_epoch = filtered_data.iloc[0]
    final_epoch = filtered_data.iloc[-1]
    
    # 计算关键指标的变化
    loss_change = ((initial_epoch['loss_sum'] - final_epoch['loss_sum']) 
                  / initial_epoch['loss_sum'] * 100)
    acc_change = (final_epoch['top100class_accuracy/all'] - initial_epoch['top100class_accuracy/all']) * 100
    
    with open(output_file, 'w') as f:
        f.write("=== 多模态BCI分类模型训练结果摘要 ===\n\n")
        f.write(f"总训练轮次: {len(filtered_data)} epochs\n\n")
        
        f.write("初始性能 (Epoch 0):\n")
        f.write(f"  总损失: {initial_epoch['loss_sum']:.4f}\n")
        f.write(f"  总体准确率 (top-100): {initial_epoch['top100class_accuracy/all']:.4f} ({initial_epoch['top100class_accuracy/all']*100:.1f}%)\n")
        f.write(f"  深度模态准确率 (top-100): {initial_epoch['top100class_accuracy/depth']:.4f} ({initial_epoch['top100class_accuracy/depth']*100:.1f}%)\n")
        f.write(f"  图像模态准确率 (top-100): {initial_epoch['top100class_accuracy/image']:.4f} ({initial_epoch['top100class_accuracy/image']*100:.1f}%)\n")
        f.write(f"  文本模态准确率 (top-100): {initial_epoch['top100class_accuracy/text']:.4f} ({initial_epoch['top100class_accuracy/text']*100:.1f}%)\n\n")
        
        f.write(f"最终性能 (Epoch {int(final_epoch['epoch'])}):\n")
        f.write(f"  总损失: {final_epoch['loss_sum']:.4f}\n")
        f.write(f"  总体准确率 (top-100): {final_epoch['top100class_accuracy/all']:.4f} ({final_epoch['top100class_accuracy/all']*100:.1f}%)\n")
        f.write(f"  深度模态准确率 (top-100): {final_epoch['top100class_accuracy/depth']:.4f} ({final_epoch['top100class_accuracy/depth']*100:.1f}%)\n")
        f.write(f"  图像模态准确率 (top-100): {final_epoch['top100class_accuracy/image']:.4f} ({final_epoch['top100class_accuracy/image']*100:.1f}%)\n")
        f.write(f"  文本模态准确率 (top-100): {final_epoch['top100class_accuracy/text']:.4f} ({final_epoch['top100class_accuracy/text']*100:.1f}%)\n\n")
        
        f.write("性能变化:\n")
        f.write(f"  损失减少: {loss_change:.2f}%\n")
        f.write(f"  准确率提升: {acc_change:.2f} 个百分点\n\n")
        
        f.write("各模态Top-5准确率 (最终epoch):\n")
        f.write(f"  深度模态 Top-5 (top-100): {final_epoch['top100class_top5_acc/depth']:.4f} ({final_epoch['top100class_top5_acc/depth']*100:.1f}%)\n")
        f.write(f"  图像模态 Top-5 (top-100): {final_epoch['top100class_top5_acc/image']:.4f} ({final_epoch['top100class_top5_acc/image']*100:.1f}%)\n")
        f.write(f"  文本模态 Top-5 (top-100): {final_epoch['top100class_top5_acc/text']:.4f} ({final_epoch['top100class_top5_acc/text']*100:.1f}%)\n\n")
        
        f.write("各种粒度的分类准确率 (最终epoch):\n")
        f.write(f"  Top-50 分类准确率: {final_epoch['top50class_accuracy/all']:.4f} ({final_epoch['top50class_accuracy/all']*100:.1f}%)\n")
        f.write(f"  Top-100 分类准确率: {final_epoch['top100class_accuracy/all']:.4f} ({final_epoch['top100class_accuracy/all']*100:.1f}%)\n")
        f.write(f"  Top-200 分类准确率: {final_epoch['top200class_accuracy/all']:.4f} ({final_epoch['top200class_accuracy/all']*100:.1f}%)\n")
    
    print(f"结果摘要已保存到: {output_file}")
def create_combined_classification_accuracy_figure(data):
    """
    创建一张图展示所有模态的分类准确率
    将不同模态（depth, image, text, all）和不同分类粒度（top-50, top-100, top-200）画在同一张图上
    """
    # 过滤只包含完整训练记录的行
    filtered_data = data[~data['epoch'].duplicated(keep='first')]
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1])
    
    # 上面一张图展示top-50/100/200中的top-100准确率（不同模态对比）
    ax_top100 = fig.add_subplot(gs[0, 0])
    
    # 下面一张图展示四种模态分别在top-50/100/200的准确率
    ax_granularity = fig.add_subplot(gs[1, 0])
    
    # 设置图表背景
    for ax in [ax_top100, ax_granularity]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    # === 顶部图: 不同模态的top-100准确率对比 ===
    modality_columns = ['top100class_accuracy/depth', 'top100class_accuracy/image', 
                       'top100class_accuracy/text', 'top100class_accuracy/all']
    modality_labels = ['Depth', 'Image', 'Text', 'All']
    modality_colors = [COLORS['depth'], COLORS['image'], COLORS['text'], COLORS['accuracy']]
    
    for i, (col, label, color) in enumerate(zip(modality_columns, modality_labels, modality_colors)):
        if col in filtered_data.columns:
            values = gaussian_filter1d(filtered_data[col], sigma=1.0)
            ax_top100.plot(filtered_data['epoch'], values, color=color, linewidth=2.5, 
                         label=f'{label} Modality')
    
    ax_top100.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_top100.set_ylabel('Accuracy (Top-100)', fontsize=12, fontweight='bold')
    ax_top100.set_ylim(0, 1.05)
    ax_top100.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_top100.set_title('Classification Accuracy by Modality (Top-100)', 
                      fontsize=14, fontweight='bold')
    ax_top100.legend(loc='lower right', fontsize=10)
    
    # === 底部图: 不同粒度下各模态的准确率 ===
    # 创建所有组合的数据结构
    modalities = ['depth', 'image', 'text', 'all']
    granularities = ['50', '100', '200']
    
    # 为了视觉区分，使用不同的线型和标记
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']
    
    # 遍历所有模态
    for i, modality in enumerate(modalities):
        # 为每个模态创建不同粒度的曲线
        for j, gran in enumerate(granularities):
            col = f'top{gran}class_accuracy/{modality}'
            if col in filtered_data.columns:
                values = gaussian_filter1d(filtered_data[col], sigma=1.0)
                ax_granularity.plot(filtered_data['epoch'], values, 
                                  color=modality_colors[i],
                                  linestyle=linestyles[i], 
                                  marker=markers[i], 
                                  markevery=5,  # 每5个点标记一次
                                  markersize=6,
                                  linewidth=2.0, 
                                  label=f'{modality_labels[i]} (Top-{gran})')
    
    ax_granularity.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_granularity.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax_granularity.set_ylim(0, 1.05)
    ax_granularity.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_granularity.set_title('Classification Accuracy by Modality and Granularity', 
                           fontsize=14, fontweight='bold')
    
    # 使用两列显示图例，使其更紧凑
    ax_granularity.legend(loc='lower right', fontsize=9, ncol=2)
    
    # 添加图表标题
    fig.suptitle("Multi-Modal Classification Performance", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

def create_combined_top5_accuracy_figure(data):
    """
    创建一张图展示所有模态的Top-5准确率
    将不同模态（depth, image, text, all）和不同分类粒度（top-50, top-100, top-200）画在同一张图上
    """
    # 过滤只包含完整训练记录的行
    filtered_data = data[~data['epoch'].duplicated(keep='first')]
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1])
    
    # 上面一张图展示top-50/100/200中的top-100的Top-5准确率（不同模态对比）
    ax_top100 = fig.add_subplot(gs[0, 0])
    
    # 下面一张图展示四种模态分别在top-50/100/200的Top-5准确率
    ax_granularity = fig.add_subplot(gs[1, 0])
    
    # 设置图表背景
    for ax in [ax_top100, ax_granularity]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    # === 顶部图: 不同模态的top-100 Top-5准确率对比 ===
    modality_columns = ['top100class_top5_acc/depth', 'top100class_top5_acc/image', 
                       'top100class_top5_acc/text', 'top100class_top5_acc/all']
    modality_labels = ['Depth', 'Image', 'Text', 'All']
    modality_colors = [COLORS['depth'], COLORS['image'], COLORS['text'], COLORS['accuracy']]
    
    for i, (col, label, color) in enumerate(zip(modality_columns, modality_labels, modality_colors)):
        if col in filtered_data.columns:
            values = gaussian_filter1d(filtered_data[col], sigma=1.0)
            ax_top100.plot(filtered_data['epoch'], values, color=color, linewidth=2.5, 
                         label=f'{label} Modality')
    
    ax_top100.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_top100.set_ylabel('Top-5 Accuracy (Top-100)', fontsize=12, fontweight='bold')
    ax_top100.set_ylim(0, 1.05)
    ax_top100.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_top100.set_title('Top-5 Accuracy by Modality (Top-100)', 
                      fontsize=14, fontweight='bold')
    ax_top100.legend(loc='lower right', fontsize=10)
    
    # === 底部图: 不同粒度下各模态的Top-5准确率 ===
    # 创建所有组合的数据结构
    modalities = ['depth', 'image', 'text', 'all']
    granularities = ['50', '100', '200']
    
    # 为了视觉区分，使用不同的线型和标记
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']
    
    # 遍历所有模态
    for i, modality in enumerate(modalities):
        # 为每个模态创建不同粒度的曲线
        for j, gran in enumerate(granularities):
            col = f'top{gran}class_top5_acc/{modality}'
            if col in filtered_data.columns:
                values = gaussian_filter1d(filtered_data[col], sigma=1.0)
                ax_granularity.plot(filtered_data['epoch'], values, 
                                  color=modality_colors[i],
                                  linestyle=linestyles[i], 
                                  marker=markers[i], 
                                  markevery=5,  # 每5个点标记一次
                                  markersize=6,
                                  linewidth=2.0, 
                                  label=f'{modality_labels[i]} (Top-{gran})')
    
    ax_granularity.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_granularity.set_ylabel('Top-5 Accuracy', fontsize=12, fontweight='bold')
    ax_granularity.set_ylim(0, 1.05)
    ax_granularity.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_granularity.set_title('Top-5 Accuracy by Modality and Granularity', 
                           fontsize=14, fontweight='bold')
    
    # 使用两列显示图例，使其更紧凑
    ax_granularity.legend(loc='lower right', fontsize=9, ncol=2)
    
    # 添加图表标题
    fig.suptitle("Multi-Modal Top-5 Classification Performance", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig
def main():
    """主函数"""
    # 从CSV文件加载数据
    try:
        data = load_data()
    except Exception as e:
        print(f"无法加载数据: {str(e)}")
        return
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    
    # 创建合并所有模态的分类准确率图表
    fig_classification = create_combined_classification_accuracy_figure(data)
    fig_classification.savefig('results/combined_classification_accuracy.png', dpi=300, bbox_inches='tight')
    fig_classification.savefig('results/combined_classification_accuracy.pdf', format='pdf', bbox_inches='tight')
    print("合并所有模态的分类准确率图表已保存为 'results/combined_classification_accuracy.png' 和 '.pdf'")
    
    # 创建合并所有模态的Top-5准确率图表
    fig_top5 = create_combined_top5_accuracy_figure(data)
    fig_top5.savefig('results/combined_top5_accuracy.png', dpi=300, bbox_inches='tight')
    fig_top5.savefig('results/combined_top5_accuracy.pdf', format='pdf', bbox_inches='tight')
    print("合并所有模态的Top-5准确率图表已保存为 'results/combined_top5_accuracy.png' 和 '.pdf'")
    
    # 以下是原来的代码，可以保留或注释掉
    # 创建训练进度图
    fig1 = create_training_progress_figure(data)
    fig1.savefig('results/bci_training_progress.png', dpi=300, bbox_inches='tight')
    fig1.savefig('results/bci_training_progress.pdf', format='pdf', bbox_inches='tight')
    print("训练进度图已保存为 'results/bci_training_progress.png' 和 '.pdf'")
    
    # 创建性能总结图
    fig2 = create_performance_summary_figure(data)
    fig2.savefig('results/bci_performance_summary.png', dpi=300, bbox_inches='tight')
    fig2.savefig('results/bci_performance_summary.pdf', format='pdf', bbox_inches='tight')
    print("性能总结图已保存为 'results/bci_performance_summary.png' 和 '.pdf'")
    
    # 保存结果摘要
    save_results_summary(data, 'results/bci_results_summary.txt')
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()