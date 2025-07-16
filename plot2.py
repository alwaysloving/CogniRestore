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

def prepare_figures():
    """设置学术论文格式的图形"""
    # 创建主图和子图
    fig = plt.figure(figsize=(22, 18), constrained_layout=True)
    gs = GridSpec(6, 2, figure=fig)
    
    # 主图 - 损失和总体准确率
    ax_main = fig.add_subplot(gs[0:2, :])
    
    # 三种模态的准确率
    ax_modality = fig.add_subplot(gs[2:4, 0])
    
    # Top-K准确率比较
    ax_topk = fig.add_subplot(gs[2:4, 1])
    
    # 不同模态Top-5准确率
    ax_top5 = fig.add_subplot(gs[4, 0])
    
    # 训练损失与验证损失
    ax_train_val = fig.add_subplot(gs[4, 1])
    
    # 最终性能横条图
    ax_bar = fig.add_subplot(gs[5, :])
    
    # 设置图表背景
    for ax in [ax_main, ax_modality, ax_topk, ax_top5, ax_train_val, ax_bar]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    return fig, ax_main, ax_modality, ax_topk, ax_top5, ax_train_val, ax_bar

def plot_main(ax, data):
    """绘制主图 - 总损失和整体准确率"""
    # 过滤只包含奇数epoch的行（完整训练记录）
    filtered_data = data[~data['epoch'].duplicated(keep='first')]
    
    # 损失曲线，使用平滑处理提高可读性
    loss = gaussian_filter1d(filtered_data['loss_sum'], sigma=1.0)
    ax.plot(filtered_data['epoch'], loss, color=COLORS['loss'], linewidth=2.5, 
            label='Total Loss')
    
    # 设置左侧Y轴（损失）
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold', color=COLORS['loss'])
    ax.tick_params(axis='y', labelcolor=COLORS['loss'])
    
    # 创建右侧Y轴（准确率）
    ax2 = ax.twinx()
    
    # top100准确率 - 所有类型
    accuracy = gaussian_filter1d(filtered_data['top100class_accuracy/all'], sigma=1.0)
    ax2.plot(filtered_data['epoch'], accuracy, color=COLORS['accuracy'], linewidth=2.5, 
             label='Overall Accuracy (top-100)')
    
    # top100准确率 - 各模态
    depth_acc = gaussian_filter1d(filtered_data['top100class_accuracy/depth'], sigma=1.0)
    image_acc = gaussian_filter1d(filtered_data['top100class_accuracy/image'], sigma=1.0)
    text_acc = gaussian_filter1d(filtered_data['top100class_accuracy/text'], sigma=1.0)
    
    ax2.plot(filtered_data['epoch'], depth_acc, color=COLORS['depth'], linewidth=1.8, 
             linestyle='--', label='Depth Accuracy (top-100)')
    ax2.plot(filtered_data['epoch'], image_acc, color=COLORS['image'], linewidth=1.8, 
             linestyle='-.', label='Image Accuracy (top-100)')
    ax2.plot(filtered_data['epoch'], text_acc, color=COLORS['text'], linewidth=1.8, 
             linestyle=':', label='Text Accuracy (top-100)')
    
    # 设置右侧Y轴（准确率）
    ax2.set_ylabel('Accuracy', fontsize=14, fontweight='bold', color=COLORS['accuracy'])
    ax2.tick_params(axis='y', labelcolor=COLORS['accuracy'])
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=12)
    
    # 设置标题
    ax.set_title('Multi-Modal (Depth, Image, Text) BCI Performance Metrics', 
                 fontsize=16, fontweight='bold', pad=20)

def plot_modality_accuracy(ax, data):
    """绘制不同模态的准确率比较"""
    # 过滤只包含奇数epoch的行（完整训练记录）
    filtered_data = data[~data['epoch'].duplicated(keep='first')]
    
    # 绘制各模态的top100准确率
    modality_columns = ['top100class_accuracy/depth', 'top100class_accuracy/image', 'top100class_accuracy/text']
    modality_labels = ['Depth', 'Image', 'Text']
    modality_colors = [COLORS['depth'], COLORS['image'], COLORS['text']]
    
    for i, col in enumerate(modality_columns):
        values = gaussian_filter1d(filtered_data[col], sigma=1.0)
        ax.plot(filtered_data['epoch'], values, color=modality_colors[i], linewidth=2, 
                label=f'{modality_labels[i]} Accuracy')
    
    # 设置轴和标题
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title('Accuracy by Modality (top-100)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=10)

def plot_topk_accuracy(ax, data):
    """绘制不同Top-K的准确率比较"""
    # 过滤只包含奇数epoch的行（完整训练记录）
    filtered_data = data[~data['epoch'].duplicated(keep='first')]
    
    # 绘制不同top-k的整体准确率
    topk_columns = ['top50class_accuracy/all', 'top100class_accuracy/all', 'top200class_accuracy/all']
    topk_labels = ['top-50', 'top-100', 'top-200']
    topk_colors = [COLORS['top50'], COLORS['top100'], COLORS['top200']]
    
    for i, col in enumerate(topk_columns):
        values = gaussian_filter1d(filtered_data[col], sigma=1.0)
        ax.plot(filtered_data['epoch'], values, color=topk_colors[i], linewidth=2, 
                label=f'{topk_labels[i]} Accuracy')
    
    # 设置轴和标题
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title('Accuracy by Classification Granularity', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=10)

def plot_top5_accuracy(ax, data):
    """绘制各模态的Top-5准确率"""
    # 过滤只包含奇数epoch的行（完整训练记录）
    filtered_data = data[~data['epoch'].duplicated(keep='first')]
    
    # 绘制各模态的top5准确率
    top5_columns = ['top100class_top5_acc/depth', 'top100class_top5_acc/image', 'top100class_top5_acc/text']
    top5_labels = ['Depth Top-5', 'Image Top-5', 'Text Top-5']
    top5_colors = [COLORS['depth'], COLORS['image'], COLORS['text']]
    
    for i, col in enumerate(top5_columns):
        values = gaussian_filter1d(filtered_data[col], sigma=1.0)
        ax.plot(filtered_data['epoch'], values, color=top5_colors[i], linewidth=2, 
                label=f'{top5_labels[i]}')
    
    # 设置轴和标题
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-5 Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title('Top-5 Accuracy by Modality (top-100)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=10)

def plot_train_val_loss(ax, data):
    """绘制训练损失与验证损失"""
    # 获取训练和验证的记录
    train_data = data[data['step'].duplicated(keep='last')]
    val_data = data[~data['epoch'].duplicated(keep='first')]
    
    # 绘制深度模态的训练和验证损失
    if 'train/loss_depth' in train_data.columns and 'val/loss_depth' in val_data.columns:
        ax.plot(train_data['epoch'], train_data['train/loss_depth'], color=COLORS['depth'], 
                linestyle='--', linewidth=1.5, label='Depth Train Loss')
        ax.plot(val_data['epoch'], val_data['val/loss_depth'], color=COLORS['depth'], 
                linewidth=2, label='Depth Val Loss')
    
    # 绘制图像模态的训练和验证损失
    if 'train/loss_img' in train_data.columns and 'val/loss_img' in val_data.columns:
        ax.plot(train_data['epoch'], train_data['train/loss_img'], color=COLORS['image'], 
                linestyle='--', linewidth=1.5, label='Image Train Loss')
        ax.plot(val_data['epoch'], val_data['val/loss_img'], color=COLORS['image'], 
                linewidth=2, label='Image Val Loss')
    
    # 绘制文本模态的训练和验证损失
    if 'train/loss_text' in train_data.columns and 'val/loss_text' in val_data.columns:
        ax.plot(train_data['epoch'], train_data['train/loss_text'], color=COLORS['text'], 
                linestyle='--', linewidth=1.5, label='Text Train Loss')
        ax.plot(val_data['epoch'], val_data['val/loss_text'], color=COLORS['text'], 
                linewidth=2, label='Text Val Loss')
    
    # 设置轴和标题
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training vs. Validation Loss by Modality', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)

def plot_final_performance(ax, data):
    """绘制最终性能横条图"""
    # 获取最后一个epoch的性能指标
    final_metrics = data[~data['epoch'].duplicated(keep='first')].iloc[-1]
    
    # 选择要展示的指标和顺序
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
        for _ in range(3):  # 每种模态3个指标(top50, top100, top200)
            colors.append(modality_colors[i])
    
    # 画横条图
    bars = ax.barh(metric_labels, values, color=colors, height=0.6, alpha=0.8)
    
    # 在条形末端添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.1%}', va='center', fontsize=10, fontweight='bold')
    
    # 设置轴和标题
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title(f'Final Performance Metrics (Epoch {int(final_metrics["epoch"])})', 
                 fontsize=14, fontweight='bold', pad=15)

def add_annotations(fig):
    """添加注释和说明"""
    # 在图的底部添加说明文字
    description = (
        "Figure 1: Training progression of multi-modal (depth, image, text) BCI classification model. "
        "The model shows differential performance across modalities, with image classification typically "
        "achieving the highest accuracy, followed by depth, and then text. "
        "Top-5 accuracy metrics demonstrate more robust performance, especially for the depth and image modalities."
    )
    
    fig.text(0.5, 0.01, description, ha='center', fontsize=12, 
             style='italic', bbox=dict(facecolor='#f0f0f0', alpha=0.5, pad=10))

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

def main():
    """主函数"""
    # 从CSV文件加载数据
    try:
        data = load_data()
    except Exception as e:
        print(f"无法加载数据: {str(e)}")
        return
    
    # 准备图形
    fig, ax_main, ax_modality, ax_topk, ax_top5, ax_train_val, ax_bar = prepare_figures()
    
    # 绘制所有子图
    plot_main(ax_main, data)
    plot_modality_accuracy(ax_modality, data)
    plot_topk_accuracy(ax_topk, data)
    plot_top5_accuracy(ax_top5, data)
    plot_train_val_loss(ax_train_val, data)
    plot_final_performance(ax_bar, data)
    
    # 添加注释和说明
    add_annotations(fig)
    
    # 保存结果摘要
    save_results_summary(data)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # 保存高分辨率图片，适合学术论文
    plt.savefig('bci_multimodal_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('bci_multimodal_results.pdf', format='pdf', bbox_inches='tight')
    
    print("图表已生成并保存为 'bci_multimodal_results.png' 和 'bci_multimodal_results.pdf'")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()