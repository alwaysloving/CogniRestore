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

def prepare_figures():
    """设置学术论文格式的图形"""
    # 创建主图和三个子图
    fig = plt.figure(figsize=(20, 16), constrained_layout=True)
    gs = GridSpec(5, 2, figure=fig)
    
    # 主图 - 损失和主准确率
    ax_main = fig.add_subplot(gs[0:2, :])
    
    # V系列准确率
    ax_v = fig.add_subplot(gs[2:4, 0])
    
    # Top-5准确率
    ax_top5 = fig.add_subplot(gs[2:4, 1])
    
    # 最终性能横条图
    ax_bar = fig.add_subplot(gs[4, :])
    
    # 设置图表背景
    for ax in [ax_main, ax_v, ax_top5, ax_bar]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    
    return fig, ax_main, ax_v, ax_top5, ax_bar

def plot_main(ax, data):
    """绘制主图 - 损失和准确率"""
    # 损失曲线，使用平滑处理提高可读性
    loss = gaussian_filter1d(data['test_loss'], sigma=1.0)
    ax.plot(data['epoch'], loss, color=COLORS['loss'], linewidth=2.5, 
            label='Test Loss')
    
    # 设置左侧Y轴（损失）
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold', color=COLORS['loss'])
    ax.tick_params(axis='y', labelcolor=COLORS['loss'])
    
    # 创建右侧Y轴（准确率）
    ax2 = ax.twinx()
    
    # 主准确率曲线
    accuracy = gaussian_filter1d(data['test_accuracy'], sigma=1.0)
    ax2.plot(data['epoch'], accuracy, color=COLORS['accuracy'], linewidth=2.5, 
             label='Test Accuracy')
    
    # 添加v2和v100准确率
    v2_acc = gaussian_filter1d(data['v2_acc'], sigma=1.0)
    v100_acc = gaussian_filter1d(data['v100_acc'], sigma=1.0)
    
    ax2.plot(data['epoch'], v2_acc, color=COLORS['v2'], linewidth=1.8, 
             linestyle='--', label='v2 Accuracy')
    ax2.plot(data['epoch'], v100_acc, color=COLORS['v100'], linewidth=1.8, 
             linestyle='-.', label='v100 Accuracy')
    
    # 设置右侧Y轴（准确率）
    ax2.set_ylabel('Accuracy', fontsize=14, fontweight='bold', color=COLORS['accuracy'])
    ax2.tick_params(axis='y', labelcolor=COLORS['accuracy'])
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=12)
    
    # 设置标题
    ax.set_title('EEG Classification/Retrieval Training Metrics', 
                 fontsize=16, fontweight='bold', pad=20)

def plot_v_accuracy(ax, data):
    """绘制V系列准确率"""
    # 绘制所有v系列准确率
    v_columns = ['v2_acc', 'v4_acc', 'v10_acc', 'v50_acc', 'v100_acc']
    v_labels = ['v2', 'v4', 'v10', 'v50', 'v100']
    v_colors = [COLORS['v2'], COLORS['v4'], COLORS['v10'], COLORS['v50'], COLORS['v100']]
    
    for i, col in enumerate(v_columns):
        values = gaussian_filter1d(data[col], sigma=1.0)
        ax.plot(data['epoch'], values, color=v_colors[i], linewidth=2, 
                label=f'{v_labels[i]} Accuracy')
    
    # 设置轴和标题
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title('Retrieval Accuracy by Candidate Set Size', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=10)

def plot_top5_accuracy(ax, data):
    """绘制Top-5准确率"""
    # 绘制所有Top-5相关准确率
    top5_columns = ['top5_acc', 'v50_top5_acc', 'v100_top5_acc']
    top5_labels = ['Top-5', 'v50 Top-5', 'v100 Top-5']
    top5_colors = [COLORS['top5'], COLORS['v50'], COLORS['v100']]
    
    for i, col in enumerate(top5_columns):
        values = gaussian_filter1d(data[col], sigma=1.0)
        ax.plot(data['epoch'], values, color=top5_colors[i], linewidth=2, 
                label=f'{top5_labels[i]} Accuracy')
    
    # 设置轴和标题
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title('Top-5 Retrieval Accuracy', fontsize=14, fontweight='bold', pad=15)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=10)

def plot_final_performance(ax, data):
    """绘制最终性能横条图"""
    # 获取最后一个epoch的性能指标
    final_metrics = data.iloc[-1]
    
    # 选择要展示的指标和顺序
    metrics = ['test_accuracy', 'v2_acc', 'v4_acc', 'v10_acc', 'v50_acc', 'v100_acc', 
               'top5_acc', 'v50_top5_acc', 'v100_top5_acc']
    metric_labels = ['Test Acc.', 'v2 Acc.', 'v4 Acc.', 'v10 Acc.', 'v50 Acc.', 'v100 Acc.',
                     'Top-5 Acc.', 'v50 Top-5', 'v100 Top-5']
    
    # 提取值
    values = [final_metrics[m] for m in metrics]
    
    # 设置颜色渐变
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(metrics)))
    
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
    ax.set_title('Final Performance Metrics (Epoch 40)', 
                 fontsize=14, fontweight='bold', pad=15)

def add_annotations(fig):
    """添加注释和说明"""
    # 在图的底部添加说明文字
    description = (
        "Figure 1: Training progression of EEG classification/retrieval model. "
        "The model shows significant improvement in retrieval capabilities across different "
        "candidate set sizes (v2-v100), with smaller candidate sets yielding higher accuracy. "
        "Top-5 metrics demonstrate robust performance even with larger retrieval pools."
    )
    
    fig.text(0.5, 0.01, description, ha='center', fontsize=12, 
             style='italic', bbox=dict(facecolor='#f0f0f0', alpha=0.5, pad=10))

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
    
    # 计算关键指标的变化
    loss_change = ((initial_epoch['test_loss'] - final_epoch['test_loss']) 
                  / initial_epoch['test_loss'] * 100)
    acc_change = (final_epoch['test_accuracy'] - initial_epoch['test_accuracy']) * 100
    
    # 查找表现最好的epoch(按准确率)
    best_epoch = data.loc[data['test_accuracy'].idxmax()]
    
    with open(output_file, 'w') as f:
        f.write("=== EEG分类/检索模型训练结果摘要 ===\n\n")
        f.write(f"总训练轮次: {len(data)} epochs\n\n")
        
        f.write("初始性能 (Epoch 1):\n")
        f.write(f"  测试损失: {initial_epoch['test_loss']:.4f}\n")
        f.write(f"  测试准确率: {initial_epoch['test_accuracy']:.4f} ({initial_epoch['test_accuracy']*100:.1f}%)\n")
        f.write(f"  v2 准确率: {initial_epoch['v2_acc']:.4f} ({initial_epoch['v2_acc']*100:.1f}%)\n")
        f.write(f"  v100 准确率: {initial_epoch['v100_acc']:.4f} ({initial_epoch['v100_acc']*100:.1f}%)\n\n")
        
        f.write("最终性能 (Epoch 40):\n")
        f.write(f"  测试损失: {final_epoch['test_loss']:.4f}\n")
        f.write(f"  测试准确率: {final_epoch['test_accuracy']:.4f} ({final_epoch['test_accuracy']*100:.1f}%)\n")
        f.write(f"  v2 准确率: {final_epoch['v2_acc']:.4f} ({final_epoch['v2_acc']*100:.1f}%)\n")
        f.write(f"  v100 准确率: {final_epoch['v100_acc']:.4f} ({final_epoch['v100_acc']*100:.1f}%)\n\n")
        
        f.write("性能变化:\n")
        f.write(f"  损失减少: {loss_change:.2f}%\n")
        f.write(f"  准确率提升: {acc_change:.2f} 个百分点\n\n")
        
        f.write("最佳性能 (准确率):\n")
        f.write(f"  Epoch: {best_epoch['epoch']}\n")
        f.write(f"  测试损失: {best_epoch['test_loss']:.4f}\n")
        f.write(f"  测试准确率: {best_epoch['test_accuracy']:.4f} ({best_epoch['test_accuracy']*100:.1f}%)\n\n")
        
        f.write("检索性能总结 (最终epoch):\n")
        for col in ['v2_acc', 'v4_acc', 'v10_acc', 'v50_acc', 'v100_acc', 
                    'top5_acc', 'v50_top5_acc', 'v100_top5_acc']:
            f.write(f"  {col}: {final_epoch[col]:.4f} ({final_epoch[col]*100:.1f}%)\n")
    
    print(f"结果摘要已保存到: {output_file}")

def main():
    """主函数"""
    # 从CSV文件加载数据
    csv_file_path = "/root/autodl-fs/EEG_Image_decode/Generation/outputs/contrast/ATMS/sub-08/04-24_17-55/ATMS_sub-08.csv"  # 更改为您的CSV文件的路径
    
    try:
        data = load_data(csv_file_path)
    except Exception as e:
        print(f"无法加载数据: {str(e)}")
        print("为了演示目的，将使用内置的样本数据...")
        
        # 如果加载失败，使用内置的样本数据作为备选
        csv_sample = """epoch,test_loss,test_accuracy,v2_acc,v4_acc,v10_acc,top5_acc,v50_acc,v100_acc,v50_top5_acc,v100_top5_acc
1,9.208568940162658,0.005,0.54,0.25,0.11,0.03,0.025,0.01,0.115,0.06
2,8.530742015838623,0.035,0.845,0.65,0.4,0.16,0.12,0.085,0.465,0.295
3,7.85393189907074,0.115,0.885,0.795,0.585,0.345,0.265,0.165,0.715,0.555
4,7.03941606760025,0.11,0.955,0.85,0.635,0.415,0.365,0.21,0.785,0.565
5,6.26093456029892,0.155,0.95,0.895,0.72,0.5,0.365,0.255,0.785,0.64
6,5.718845069408417,0.195,0.94,0.875,0.73,0.55,0.44,0.3,0.815,0.7
7,5.288426914215088,0.235,0.98,0.895,0.76,0.575,0.45,0.325,0.86,0.735
8,4.901444118022919,0.255,0.965,0.895,0.745,0.6,0.465,0.35,0.865,0.775
9,4.55804114818573,0.265,0.955,0.91,0.77,0.635,0.475,0.365,0.895,0.77
10,4.244906537532806,0.27,0.955,0.87,0.795,0.605,0.505,0.355,0.89,0.75
11,3.9576597702503205,0.305,0.975,0.9,0.82,0.64,0.51,0.425,0.905,0.785
12,3.733158166408539,0.305,0.955,0.905,0.825,0.65,0.515,0.38,0.9,0.795
13,3.479783396720886,0.33,0.98,0.93,0.8,0.69,0.54,0.41,0.865,0.77
14,3.2726083540916444,0.335,0.965,0.89,0.79,0.68,0.52,0.435,0.915,0.8
15,3.0344137060642242,0.335,0.975,0.91,0.835,0.67,0.55,0.41,0.895,0.775
16,2.8708697831630707,0.34,0.955,0.92,0.83,0.67,0.52,0.425,0.885,0.815
17,2.689907395839691,0.35,0.955,0.935,0.82,0.705,0.6,0.475,0.91,0.83
18,2.492464655637741,0.345,0.985,0.93,0.78,0.715,0.525,0.47,0.915,0.815
19,2.2938496911525728,0.355,0.97,0.915,0.85,0.695,0.605,0.45,0.91,0.81
20,1.7386526864767076,0.31,0.93,0.91,0.845,0.625,0.545,0.425,0.9,0.77
21,0.9790234673023224,0.295,0.965,0.905,0.795,0.64,0.48,0.37,0.885,0.77
22,0.7499204289913177,0.32,0.955,0.895,0.76,0.635,0.515,0.42,0.895,0.775
23,0.6715526540577411,0.345,0.97,0.905,0.79,0.675,0.56,0.465,0.92,0.795
24,0.5469625084102154,0.355,0.97,0.915,0.8,0.685,0.59,0.43,0.89,0.79
25,0.536244934797287,0.37,0.965,0.93,0.845,0.715,0.62,0.465,0.94,0.83
26,0.4488375872373581,0.355,0.98,0.915,0.855,0.735,0.585,0.495,0.925,0.85
27,0.45132323533296587,0.36,0.99,0.925,0.88,0.735,0.59,0.435,0.925,0.815
28,0.44517708390951155,0.39,0.965,0.945,0.88,0.735,0.65,0.49,0.91,0.84
29,0.4348779910057783,0.375,0.98,0.925,0.825,0.7,0.6,0.5,0.935,0.855
30,0.46678010135889053,0.385,0.965,0.945,0.83,0.73,0.615,0.485,0.915,0.835
31,0.4509107758104801,0.37,0.985,0.91,0.875,0.725,0.595,0.465,0.925,0.845
32,0.4723525992035866,0.39,0.98,0.91,0.85,0.715,0.58,0.505,0.945,0.84
33,0.45795791685581205,0.4,0.97,0.955,0.83,0.735,0.615,0.51,0.92,0.87
34,0.48793601125478747,0.39,0.98,0.945,0.88,0.74,0.64,0.455,0.93,0.83
35,0.5508740028738975,0.395,0.98,0.9,0.84,0.745,0.62,0.56,0.945,0.87
36,0.47331392258405686,0.39,0.97,0.9,0.84,0.74,0.615,0.52,0.94,0.85
37,0.5079199443757534,0.39,0.965,0.92,0.88,0.76,0.59,0.495,0.925,0.855
38,0.5423079583048821,0.395,0.975,0.95,0.86,0.71,0.595,0.505,0.935,0.855
39,0.5671221713721752,0.41,0.975,0.955,0.88,0.73,0.615,0.53,0.945,0.855
40,0.5807751096785069,0.405,0.98,0.945,0.885,0.74,0.625,0.515,0.935,0.84"""
        
        data = pd.read_csv(pd.StringIO(csv_sample))
        print(f"已加载内置的样本数据，包含 {len(data)} 条记录")
    
    # 准备图形
    fig, ax_main, ax_v, ax_top5, ax_bar = prepare_figures()
    
    # 绘制所有子图
    plot_main(ax_main, data)
    plot_v_accuracy(ax_v, data)
    plot_top5_accuracy(ax_top5, data)
    plot_final_performance(ax_bar, data)
    
    # 添加注释和说明
    add_annotations(fig)
    
    # 保存结果摘要
    save_results_summary(data)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # 保存高分辨率图片，适合学术论文
    plt.savefig('eeg_classification_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('eeg_classification_results.pdf', format='pdf', bbox_inches='tight')
    
    print("图表已生成并保存为 'eeg_classification_results.png' 和 'eeg_classification_results.pdf'")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()