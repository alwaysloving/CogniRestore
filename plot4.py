import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import os

# 设置学术风格的可视化参数（使用更通用的样式设置，不依赖特定的样式文件）
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
    'class50': '#2ca02c',      # 绿色
    'class100': '#ff7f0e',     # 橙色
    'class200': '#9467bd',     # 紫色
    'v2': '#2ca02c',           # 绿色
    'v4': '#ff7f0e',           # 橙色
    'v10': '#9467bd',          # 紫色
    'v50': '#8c564b',          # 棕色
    'v100': '#e377c2',         # 粉色
    'top5': '#7f7f7f',         # 灰色
    'grid': '#cccccc',         # 网格线颜色
    'background': '#f9f9f9'    # 背景色
}

def load_data(file_path):
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
        required_columns = ['epoch', 'test_loss', 'test_accuracy']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要的列: {', '.join(missing_columns)}")
            
        print(f"成功从 {file_path} 加载了 {len(data)} 条训练记录")
        return data
        
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

def create_classification_comparison_figure(data):
    """创建分类任务准确率对比图"""
    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, figure=fig)
    
    # 顶部：分类准确率对比曲线
    ax_acc = fig.add_subplot(gs[0, 0])
    
    # 底部：分类Top-5准确率对比曲线
    ax_top5 = fig.add_subplot(gs[1, 0])
    
    # 设置图表背景
    for ax in [ax_acc, ax_top5]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    # === 顶部图: 不同规模的分类准确率对比 ===
    # 准确率曲线
    class50_acc = gaussian_filter1d(data['class50_accuracy'], sigma=1.0)
    class100_acc = gaussian_filter1d(data['class100_accuracy'], sigma=1.0)
    class200_acc = gaussian_filter1d(data['test_accuracy'], sigma=1.0)  # test_accuracy实际上是200类准确率
    
    ax_acc.plot(data['epoch'], class50_acc, color=COLORS['class50'], linewidth=2.5, 
                label='50-class Accuracy')
    ax_acc.plot(data['epoch'], class100_acc, color=COLORS['class100'], linewidth=2.5, 
                label='100-class Accuracy')
    ax_acc.plot(data['epoch'], class200_acc, color=COLORS['class200'], linewidth=2.5, 
                label='200-class Accuracy')
    
    ax_acc.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_acc.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax_acc.set_ylim(0, 1.05)
    ax_acc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_acc.set_title('Classification Accuracy at Different Scales', 
                   fontsize=14, fontweight='bold')
    ax_acc.legend(loc='lower right', fontsize=10)
    
    # === 底部图: 不同规模的Top-5准确率对比 ===
    # Top-5准确率曲线
    class50_top5 = gaussian_filter1d(data['class50_top5_acc'], sigma=1.0)
    class100_top5 = gaussian_filter1d(data['class100_top5_acc'], sigma=1.0)
    class200_top5 = gaussian_filter1d(data['top5_acc'], sigma=1.0)  # top5_acc实际上是200类Top-5准确率
    
    ax_top5.plot(data['epoch'], class50_top5, color=COLORS['class50'], linewidth=2.5, 
                 label='50-class Top-5 Accuracy')
    ax_top5.plot(data['epoch'], class100_top5, color=COLORS['class100'], linewidth=2.5, 
                 label='100-class Top-5 Accuracy')
    ax_top5.plot(data['epoch'], class200_top5, color=COLORS['class200'], linewidth=2.5, 
                 label='200-class Top-5 Accuracy')
    
    ax_top5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_top5.set_ylabel('Top-5 Accuracy', fontsize=12, fontweight='bold')
    ax_top5.set_ylim(0, 1.05)
    ax_top5.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_top5.set_title('Top-5 Classification Accuracy at Different Scales', 
                      fontsize=14, fontweight='bold')
    ax_top5.legend(loc='lower right', fontsize=10)
    
    # 添加图表标题和说明
    fig.suptitle("Different Scale Classification Performance Comparison", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    description = (
        "Figure 1: Comparison of classification accuracy at different scales (50, 100, 200 classes). "
        "Accuracy decreases as the number of classes increases, while Top-5 accuracy shows more robust performance "
        "even with a larger number of classes."
    )
    
    fig.text(0.5, 0.01, description, ha='center', fontsize=12, 
             style='italic', bbox=dict(facecolor='#f0f0f0', alpha=0.5, pad=10))
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.12)
    
    return fig

def create_training_progress_figure(data):
    """创建训练进度综合图表"""
    # 创建图形和子图布局
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # 顶部大图: 损失和总体准确率
    ax_main = fig.add_subplot(gs[0, :])
    
    # 左下: v系列准确率
    ax_v = fig.add_subplot(gs[1, 0])
    
    # 右下: Top-5准确率
    ax_top5 = fig.add_subplot(gs[1, 1])
    
    # 底部: 比较v2和v100的准确率曲线
    ax_compare = fig.add_subplot(gs[2, :])
    
    # 设置图表背景
    for ax in [ax_main, ax_v, ax_top5, ax_compare]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    # === 主图: 损失和测试准确率 ===
    # 损失曲线
    loss = gaussian_filter1d(data['test_loss'], sigma=1.0)
    ax_main.plot(data['epoch'], loss, color=COLORS['loss'], linewidth=2.5, 
                 label='Test Loss')
    
    ax_main.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Loss', fontsize=12, fontweight='bold', color=COLORS['loss'])
    ax_main.tick_params(axis='y', labelcolor=COLORS['loss'])
    
    # 右侧轴: 准确率
    ax_main2 = ax_main.twinx()
    
    # 测试准确率
    accuracy = gaussian_filter1d(data['test_accuracy'], sigma=1.0)
    ax_main2.plot(data['epoch'], accuracy, color=COLORS['accuracy'], linewidth=2.5, 
                  label='Test Accuracy')
    
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
    
    ax_main.set_title('EEG Classification/Retrieval Training Progress', 
                      fontsize=14, fontweight='bold')
    
    # === 左下图: v系列准确率 ===
    # 绘制v系列准确率
    v_columns = ['v2_acc', 'v4_acc', 'v10_acc', 'v50_acc', 'v100_acc']
    v_labels = ['v2', 'v4', 'v10', 'v50', 'v100']
    v_colors = [COLORS['v2'], COLORS['v4'], COLORS['v10'], COLORS['v50'], COLORS['v100']]
    
    for i, col in enumerate(v_columns):
        values = gaussian_filter1d(data[col], sigma=1.0)
        ax_v.plot(data['epoch'], values, color=v_colors[i], linewidth=2, 
                  label=f'{v_labels[i]} Accuracy')
    
    ax_v.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax_v.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
    ax_v.set_ylim(0, 1.05)
    ax_v.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_v.set_title('Retrieval Accuracy by Candidate Set Size', 
                   fontsize=12, fontweight='bold')
    ax_v.legend(loc='lower right', fontsize=9)
    
    # === 右下图: Top-5准确率 ===
    # 绘制Top-5准确率
    top5_columns = ['top5_acc', 'v50_top5_acc', 'v100_top5_acc']
    top5_labels = ['Top-5', 'v50 Top-5', 'v100 Top-5']
    top5_colors = [COLORS['top5'], COLORS['v50'], COLORS['v100']]
    
    for i, col in enumerate(top5_columns):
        values = gaussian_filter1d(data[col], sigma=1.0)
        ax_top5.plot(data['epoch'], values, color=top5_colors[i], linewidth=2, 
                     label=f'{top5_labels[i]}')
    
    ax_top5.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax_top5.set_ylabel('Top-5 Accuracy', fontsize=10, fontweight='bold')
    ax_top5.set_ylim(0, 1.05)
    ax_top5.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_top5.set_title('Top-5 Retrieval Accuracy', 
                      fontsize=12, fontweight='bold')
    ax_top5.legend(loc='lower right', fontsize=9)
    
    # === 底部图: v2和v100的对比 ===
    # 绘制v2和v100准确率的对比曲线
    v2_acc = gaussian_filter1d(data['v2_acc'], sigma=1.0)
    v100_acc = gaussian_filter1d(data['v100_acc'], sigma=1.0)
    v2_top5 = gaussian_filter1d(data['top5_acc'], sigma=1.0)  # Using top5_acc as a proxy for v2 top5
    v100_top5 = gaussian_filter1d(data['v100_top5_acc'], sigma=1.0)
    
    ax_compare.plot(data['epoch'], v2_acc, color=COLORS['v2'], linewidth=2, 
                    label='v2 Accuracy')
    ax_compare.plot(data['epoch'], v100_acc, color=COLORS['v100'], linewidth=2, 
                    label='v100 Accuracy')
    ax_compare.plot(data['epoch'], v2_top5, color=COLORS['v2'], linewidth=2, 
                    linestyle='--', label='v2 Top-5 proxy')
    ax_compare.plot(data['epoch'], v100_top5, color=COLORS['v100'], linewidth=2, 
                    linestyle='--', label='v100 Top-5')
    
    ax_compare.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax_compare.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
    ax_compare.set_ylim(0, 1.05)
    ax_compare.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_compare.set_title('Comparison of Small vs Large Retrieval Set Performance', 
                         fontsize=12, fontweight='bold')
    ax_compare.legend(loc='lower right', fontsize=9)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def create_performance_summary_figure(data):
    """创建性能总结图表"""
    # 获取最后一个epoch的记录
    final_metrics = data.iloc[-1]
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 顶部：最终性能横条图
    ax_bar = fig.add_subplot(gs[0, :])
    
    # 左下：准确率比较
    ax_acc = fig.add_subplot(gs[1, 0])
    
    # 右下：最终数值摘要
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis('off')  # 隐藏坐标轴
    
    # 设置图表背景
    for ax in [ax_bar, ax_acc]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    # === 顶部：最终性能横条图 ===
    metrics = ['test_accuracy', 'class50_accuracy', 'class100_accuracy', 
               'v2_acc', 'v4_acc', 'v10_acc', 'v50_acc', 'v100_acc', 
               'top5_acc', 'class50_top5_acc', 'class100_top5_acc', 'v50_top5_acc', 'v100_top5_acc']
    metric_labels = ['200-class Acc.', '50-class Acc.', '100-class Acc.',
                     'v2 Acc.', 'v4 Acc.', 'v10 Acc.', 'v50 Acc.', 'v100 Acc.',
                     '200-class Top-5', '50-class Top-5', '100-class Top-5', 'v50 Top-5', 'v100 Top-5']
    
    # 提取值
    values = [final_metrics[m] for m in metrics]
    
    # 设置颜色渐变
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(metrics)))
    
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
    
    # === 左下图：准确率增长对比 ===
    # 计算每个指标随epoch的提升百分比
    initial_epoch = data.iloc[0]
    improvement_data = []
    
    # 计算测试准确率和主要v系列准确率的提升
    key_metrics = ['test_accuracy', 'class50_accuracy', 'class100_accuracy', 'v2_acc', 'v100_acc']
    key_labels = ['200-class Acc.', '50-class Acc.', '100-class Acc.', 'v2 Acc.', 'v100 Acc.']
    key_colors = [COLORS['class200'], COLORS['class50'], COLORS['class100'], COLORS['v2'], COLORS['v100']]
    
    for epoch_idx in range(0, len(data), max(1, len(data) // 5)):  # 取5个时间点
        epoch_data = data.iloc[epoch_idx]
        improvement = {}
        improvement['epoch'] = epoch_data['epoch']
        
        for metric in key_metrics:
            if initial_epoch[metric] > 0:
                improvement[metric] = epoch_data[metric] / initial_epoch[metric]
            else:
                improvement[metric] = epoch_data[metric] * 100 if epoch_data[metric] > 0 else 1.0
        
        improvement_data.append(improvement)
    
    # 创建准确率提升数据
    improvement_df = pd.DataFrame(improvement_data)
    
    # 宽度
    width = 0.18  # 由于增加了指标数量，缩小宽度
    x = np.arange(len(improvement_df))
    
    # 绘制分组条形图
    for i, metric in enumerate(key_metrics):
        ax_acc.bar(x + i*width, improvement_df[metric], width, 
                   label=key_labels[i], color=key_colors[i], alpha=0.8)
    
    # 设置X轴标签为epoch
    ax_acc.set_xticks(x + width * (len(key_metrics) - 1) / 2)
    ax_acc.set_xticklabels([f"Epoch {int(e)}" for e in improvement_df['epoch']])
    
    ax_acc.set_ylabel('Relative Improvement (x times)', fontsize=10, fontweight='bold')
    ax_acc.set_title('Accuracy Improvement Over Training', 
                     fontsize=12, fontweight='bold')
    ax_acc.legend(loc='upper left', fontsize=8)
    
    # === 右下图：结果摘要文本 ===
    # 计算关键指标的变化
    initial_epoch = data.iloc[0]
    loss_change = ((initial_epoch['test_loss'] - final_metrics['test_loss']) 
                  / initial_epoch['test_loss'] * 100)
    acc_change = (final_metrics['test_accuracy'] - initial_epoch['test_accuracy']) * 100
    class50_acc_change = (final_metrics['class50_accuracy'] - initial_epoch['class50_accuracy']) * 100
    class100_acc_change = (final_metrics['class100_accuracy'] - initial_epoch['class100_accuracy']) * 100
    
    # 查找表现最好的epoch(按准确率)
    best_epoch = data.loc[data['test_accuracy'].idxmax()]
    
    # 创建结果摘要文本
    summary_text = (
        f"TRAINING SUMMARY (Epochs: {len(data)})\n\n"
        f"LOSS REDUCTION:\n"
        f"  Initial: {initial_epoch['test_loss']:.4f}\n"
        f"  Final: {final_metrics['test_loss']:.4f}\n"
        f"  Change: -{loss_change:.1f}%\n\n"
        f"CLASSIFICATION IMPROVEMENT:\n"
        f"  50-class: +{class50_acc_change:.1f} points\n"
        f"  100-class: +{class100_acc_change:.1f} points\n"
        f"  200-class: +{acc_change:.1f} points\n\n"
        f"BEST PERFORMANCE (Epoch {int(best_epoch['epoch'])}):\n"
        f"  200-class Acc: {best_epoch['test_accuracy']*100:.1f}%\n\n"
        f"RETRIEVAL PERFORMANCE (Final):\n"
        f"  v2: {final_metrics['v2_acc']*100:.1f}%\n"
        f"  v100: {final_metrics['v100_acc']*100:.1f}%\n"
        f"  v100 Top-5: {final_metrics['v100_top5_acc']*100:.1f}%"
    )
    
    # 添加摘要文本
    ax_text.text(0.05, 0.95, summary_text, va='top', fontsize=12, 
                 backgroundcolor=COLORS['background'], 
                 bbox=dict(facecolor=COLORS['background'], edgecolor='#cccccc', 
                           boxstyle='round,pad=1', alpha=0.9))
    
    # 添加图标题
    fig.suptitle("EEG Classification/Retrieval Model Performance Summary", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 添加说明文字
    description = (
        "Figure 3: Performance summary of EEG classification/retrieval model. "
        "The model shows differential performance across classification scales (50, 100, 200 classes) "
        "and retrieval set sizes (v2-v100), with smaller scales and sets yielding higher accuracy."
    )
    
    fig.text(0.5, 0.01, description, ha='center', fontsize=12, 
             style='italic', bbox=dict(facecolor='#f0f0f0', alpha=0.5, pad=10))
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08)
    
    return fig

def save_results_summary(data, output_file='eeg_results_summary.txt'):
    """
    保存训练结果的摘要统计信息到文本文件
    
    参数:
        data: 训练数据的DataFrame
        output_file: 输出文件的路径
    """
    # 获取初始和最终epoch的数据
    initial_epoch = data.iloc[0]
    final_epoch = data.iloc[-1]
    
    # 计算关键指标的变化 - 修复变量名称错误
    loss_change = ((initial_epoch['test_loss'] - final_epoch['test_loss']) 
                  / initial_epoch['test_loss'] * 100)
    acc_change = (final_epoch['test_accuracy'] - initial_epoch['test_accuracy']) * 100
    class50_acc_change = (final_epoch['class50_accuracy'] - initial_epoch['class50_accuracy']) * 100
    class100_acc_change = (final_epoch['class100_accuracy'] - initial_epoch['class100_accuracy']) * 100
    
    # 查找表现最好的epoch(按准确率)
    best_epoch = data.loc[data['test_accuracy'].idxmax()]
    
    with open(output_file, 'w') as f:
        f.write("=== EEG分类/检索模型训练结果摘要 ===\n\n")
        f.write(f"总训练轮次: {len(data)} epochs\n\n")
        
        f.write("初始性能 (Epoch 1):\n")
        f.write(f"  测试损失: {initial_epoch['test_loss']:.4f}\n")
        f.write(f"  50类准确率: {initial_epoch['class50_accuracy']:.4f} ({initial_epoch['class50_accuracy']*100:.1f}%)\n")
        f.write(f"  100类准确率: {initial_epoch['class100_accuracy']:.4f} ({initial_epoch['class100_accuracy']*100:.1f}%)\n")
        f.write(f"  200类准确率: {initial_epoch['test_accuracy']:.4f} ({initial_epoch['test_accuracy']*100:.1f}%)\n")
        f.write(f"  v2 检索准确率: {initial_epoch['v2_acc']:.4f} ({initial_epoch['v2_acc']*100:.1f}%)\n")
        f.write(f"  v100 检索准确率: {initial_epoch['v100_acc']:.4f} ({initial_epoch['v100_acc']*100:.1f}%)\n\n")
        
        f.write(f"最终性能 (Epoch {int(final_epoch['epoch'])}):\n")
        f.write(f"  测试损失: {final_epoch['test_loss']:.4f}\n")
        f.write(f"  50类准确率: {final_epoch['class50_accuracy']:.4f} ({final_epoch['class50_accuracy']*100:.1f}%)\n")
        f.write(f"  100类准确率: {final_epoch['class100_accuracy']:.4f} ({final_epoch['class100_accuracy']*100:.1f}%)\n")
        f.write(f"  200类准确率: {final_epoch['test_accuracy']:.4f} ({final_epoch['test_accuracy']*100:.1f}%)\n")
        f.write(f"  v2 检索准确率: {final_epoch['v2_acc']:.4f} ({final_epoch['v2_acc']*100:.1f}%)\n")
        f.write(f"  v100 检索准确率: {final_epoch['v100_acc']:.4f} ({final_epoch['v100_acc']*100:.1f}%)\n\n")
        
        f.write("性能变化:\n")
        f.write(f"  损失减少: {loss_change:.2f}%\n")
        f.write(f"  50类准确率提升: {class50_acc_change:.2f} 个百分点\n")
        f.write(f"  100类准确率提升: {class100_acc_change:.2f} 个百分点\n")
        f.write(f"  200类准确率提升: {acc_change:.2f} 个百分点\n\n")
        
        f.write("最佳性能 (200类准确率):\n")
        f.write(f"  Epoch: {int(best_epoch['epoch'])}\n")
        f.write(f"  测试损失: {best_epoch['test_loss']:.4f}\n")
        f.write(f"  200类准确率: {best_epoch['test_accuracy']:.4f} ({best_epoch['test_accuracy']*100:.1f}%)\n\n")
        
        f.write("分类性能总结 (最终epoch):\n")
        for col in ['class50_accuracy', 'class100_accuracy', 'test_accuracy', 
                    'class50_top5_acc', 'class100_top5_acc', 'top5_acc']:
            f.write(f"  {col}: {final_epoch[col]:.4f} ({final_epoch[col]*100:.1f}%)\n")
        
        f.write("\n检索性能总结 (最终epoch):\n")
        for col in ['v2_acc', 'v4_acc', 'v10_acc', 'v50_acc', 'v100_acc', 
                    'v50_top5_acc', 'v100_top5_acc']:
            f.write(f"  {col}: {final_epoch[col]:.4f} ({final_epoch[col]*100:.1f}%)\n")
    
    print(f"结果摘要已保存到: {output_file}")
def create_classification_only_figure(data):
    """创建仅展示分类结果的图表"""
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(2, 1, figure=fig)
    
    # 顶部：分类准确率曲线
    ax_acc = fig.add_subplot(gs[0, 0])
    
    # 底部：分类Top-5准确率曲线
    ax_top5 = fig.add_subplot(gs[1, 0])
    
    # 设置图表背景
    for ax in [ax_acc, ax_top5]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    # === 顶部图: 不同规模的分类准确率对比 ===
    # 准确率曲线
    class50_acc = gaussian_filter1d(data['class50_accuracy'], sigma=1.0)
    class100_acc = gaussian_filter1d(data['class100_accuracy'], sigma=1.0)
    class200_acc = gaussian_filter1d(data['test_accuracy'], sigma=1.0)  # test_accuracy实际上是200类准确率
    
    ax_acc.plot(data['epoch'], class50_acc, color=COLORS['class50'], linewidth=2.5, 
                label='50-class Accuracy')
    ax_acc.plot(data['epoch'], class100_acc, color=COLORS['class100'], linewidth=2.5, 
                label='100-class Accuracy')
    ax_acc.plot(data['epoch'], class200_acc, color=COLORS['class200'], linewidth=2.5, 
                label='200-class Accuracy')
    
    ax_acc.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_acc.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax_acc.set_ylim(0, 1.05)
    ax_acc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_acc.set_title('Classification Accuracy at Different Scales', 
                   fontsize=14, fontweight='bold')
    ax_acc.legend(loc='lower right', fontsize=10)
    
    # === 底部图: 不同规模的Top-5准确率对比 ===
    # Top-5准确率曲线
    class50_top5 = gaussian_filter1d(data['class50_top5_acc'], sigma=1.0)
    class100_top5 = gaussian_filter1d(data['class100_top5_acc'], sigma=1.0)
    class200_top5 = gaussian_filter1d(data['top5_acc'], sigma=1.0)  # top5_acc实际上是200类Top-5准确率
    
    ax_top5.plot(data['epoch'], class50_top5, color=COLORS['class50'], linewidth=2.5, 
                 label='50-class Top-5 Accuracy')
    ax_top5.plot(data['epoch'], class100_top5, color=COLORS['class100'], linewidth=2.5, 
                 label='100-class Top-5 Accuracy')
    ax_top5.plot(data['epoch'], class200_top5, color=COLORS['class200'], linewidth=2.5, 
                 label='200-class Top-5 Accuracy')
    
    ax_top5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_top5.set_ylabel('Top-5 Accuracy', fontsize=12, fontweight='bold')
    ax_top5.set_ylim(0, 1.05)
    ax_top5.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_top5.set_title('Top-5 Classification Accuracy at Different Scales', 
                      fontsize=14, fontweight='bold')
    ax_top5.legend(loc='lower right', fontsize=10)
    
    # 添加图表标题
    fig.suptitle("Classification Performance Comparison", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

def create_retrieval_only_figure(data):
    """创建仅展示检索结果的图表"""
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(2, 1, figure=fig)
    
    # 顶部：检索准确率曲线
    ax_ret = fig.add_subplot(gs[0, 0])
    
    # 底部：检索Top-5准确率曲线
    ax_ret_top5 = fig.add_subplot(gs[1, 0])
    
    # 设置图表背景
    for ax in [ax_ret, ax_ret_top5]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    # === 顶部图: v系列检索准确率 ===
    # 绘制v系列准确率
    v_columns = ['v2_acc', 'v4_acc', 'v10_acc', 'v50_acc', 'v100_acc']
    v_labels = ['v2', 'v4', 'v10', 'v50', 'v100']
    v_colors = [COLORS['v2'], COLORS['v4'], COLORS['v10'], COLORS['v50'], COLORS['v100']]
    
    for i, col in enumerate(v_columns):
        values = gaussian_filter1d(data[col], sigma=1.0)
        ax_ret.plot(data['epoch'], values, color=v_colors[i], linewidth=2.5, 
                  label=f'{v_labels[i]} Accuracy')
    
    ax_ret.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_ret.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax_ret.set_ylim(0, 1.05)
    ax_ret.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_ret.set_title('Retrieval Accuracy by Candidate Set Size', 
                   fontsize=14, fontweight='bold')
    ax_ret.legend(loc='lower right', fontsize=10)
    
    # === 底部图: Top-5检索准确率 ===
    # 绘制Top-5检索准确率
    top5_columns = ['top5_acc', 'v50_top5_acc', 'v100_top5_acc']
    top5_labels = ['Global Top-5', 'v50 Top-5', 'v100 Top-5']
    top5_colors = [COLORS['top5'], COLORS['v50'], COLORS['v100']]
    
    for i, col in enumerate(top5_columns):
        values = gaussian_filter1d(data[col], sigma=1.0)
        ax_ret_top5.plot(data['epoch'], values, color=top5_colors[i], linewidth=2.5, 
                     label=f'{top5_labels[i]}')
    
    ax_ret_top5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_ret_top5.set_ylabel('Top-5 Accuracy', fontsize=12, fontweight='bold')
    ax_ret_top5.set_ylim(0, 1.05)
    ax_ret_top5.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_ret_top5.set_title('Top-5 Retrieval Accuracy', 
                      fontsize=14, fontweight='bold')
    ax_ret_top5.legend(loc='lower right', fontsize=10)
    
    # 添加图表标题
    fig.suptitle("Retrieval Performance Comparison", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

def main():
    """主函数"""
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    
    # 从CSV文件加载数据
    csv_file_path = "/root/autodl-fs/EEG_Image_decode/Generation/outputs/contrast/ATMS/sub-08/05-08_01-31/ATMS_sub-08.csv"  # 默认文件名，用户可以修改
    
    try:
        data = load_data(csv_file_path)
    except Exception as e:
        print(f"无法加载数据: {str(e)}")
        
        # 使用用户提供的数据作为备选
        csv_data = """epoch,test_loss,test_accuracy,class50_accuracy,class100_accuracy,v2_acc,v4_acc,v10_acc,top5_acc,v50_acc,v100_acc,v50_top5_acc,v100_top5_acc,class50_top5_acc,class100_top5_acc
1,8.545202789306641,0.04,0.11,0.07,0.78,0.565,0.365,0.15,0.125,0.07,0.385,0.255,0.425,0.225
2,7.2772297143936155,0.115,0.26,0.175,0.895,0.79,0.56,0.31,0.285,0.175,0.655,0.52,0.635,0.52
...
40,0.5766511280834675,0.44,0.58,0.53,0.98,0.94,0.87,0.73,0.63,0.535,0.935,0.84,0.95,0.865"""
        
        # 使用StringIO将字符串解析为DataFrame
        data = pd.read_csv(pd.StringIO(csv_data))
        print(f"已加载内置的样本数据，包含 {len(data)} 条记录")
    
    # 创建分类任务准确率图表 (新增)
    fig_class = create_classification_only_figure(data)
    fig_class.savefig('results/classification_only.png', dpi=300, bbox_inches='tight')
    fig_class.savefig('results/classification_only.pdf', format='pdf', bbox_inches='tight')
    print("分类准确率图表已保存为 'results/classification_only.png' 和 'results/classification_only.pdf'")
    
    # 创建检索任务准确率图表 (新增)
    fig_ret = create_retrieval_only_figure(data)
    fig_ret.savefig('results/retrieval_only.png', dpi=300, bbox_inches='tight')
    fig_ret.savefig('results/retrieval_only.pdf', format='pdf', bbox_inches='tight')
    print("检索准确率图表已保存为 'results/retrieval_only.png' 和 'results/retrieval_only.pdf'")
    
    # 以下是原来的代码，可以保留或注释掉
    # 创建不同规模的分类任务准确率对比图
    fig1 = create_classification_comparison_figure(data)
    fig1.savefig('results/classification_comparison.png', dpi=300, bbox_inches='tight')
    fig1.savefig('results/classification_comparison.pdf', format='pdf', bbox_inches='tight')
    print("分类任务对比图已保存为 'results/classification_comparison.png' 和 'results/classification_comparison.pdf'")
    
    # 创建训练进度图
    fig2 = create_training_progress_figure(data)
    fig2.savefig('results/eeg_training_progress.png', dpi=300, bbox_inches='tight')
    fig2.savefig('results/eeg_training_progress.pdf', format='pdf', bbox_inches='tight')
    print("训练进度图已保存为 'results/eeg_training_progress.png' 和 'results/eeg_training_progress.pdf'")
    
    # 创建性能总结图
    fig3 = create_performance_summary_figure(data)
    fig3.savefig('results/eeg_performance_summary.png', dpi=300, bbox_inches='tight')
    fig3.savefig('results/eeg_performance_summary.pdf', format='pdf', bbox_inches='tight')
    print("性能总结图已保存为 'results/eeg_performance_summary.png' 和 'results/eeg_performance_summary.pdf'")
    
    # 保存结果摘要
    save_results_summary(data, 'results/eeg_results_summary.txt')
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()