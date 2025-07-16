import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class FrequencyFeatureExtractor:
    def __init__(self, device='cuda:0'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
    def extract_frequency_features(self, image):
        """从图像中提取频域特征"""
        # 确保图像是正确的数据类型
        if image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        elif image.dtype == np.float32 and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
        # 转换为灰度图
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
            
        # 应用傅里叶变换
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        
        # 计算幅度谱
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # 提取不同频段的特征
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # 定义频段区域
        low_freq = magnitude_spectrum[center_h-10:center_h+10, center_w-10:center_w+10]
        mid_freq = magnitude_spectrum[center_h-30:center_h+30, center_w-30:center_w+30]
        high_freq = magnitude_spectrum
        
        # 计算各频段的能量
        low_energy = np.mean(low_freq)
        mid_energy = np.mean(mid_freq)
        high_energy = np.mean(high_freq)
        
        # 组合频域特征
        frequency_features = {
            'magnitude_spectrum': magnitude_spectrum,
            'low_freq_energy': low_energy,
            'mid_freq_energy': mid_energy,
            'high_freq_energy': high_energy
        }
        
        return frequency_features

def get_frequency_extractor():
    """获取频域特征提取器"""
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    extractor = FrequencyFeatureExtractor(DEVICE)
    return extractor, DEVICE

def process_single_image(args):
    """处理单张图像的函数，用于多进程处理"""
    img_path, extractor = args
    return get_frequency_features(img_path, extractor, None)

def img_frequency():
    """提取图像的频域特征"""
    saved_path = "/root/autodl-fs/CognitionCapturer/image_set/img_path/"
    extractor, DEVICE = get_frequency_extractor()
    
    # 使用多进程处理
    num_workers = min(multiprocessing.cpu_count() - 1, 8)  # 留一个CPU核心
    
    for i, file_name in enumerate(['train_image.npy', 'test_image.npy']):
        file_path = os.path.join(saved_path, file_name)
        paths = np.load(file_path, allow_pickle=True).tolist()
        
        # 创建参数列表
        args_list = [(img_path, extractor) for img_path in paths]
        
        # 使用多进程池处理图像
        with multiprocessing.Pool(num_workers) as pool:
            list(tqdm(pool.imap(process_single_image, args_list), 
                     total=len(paths), 
                     desc=f"Processing {file_name}"))

def batch_process_images(img_paths, extractor, batch_size=1024):
    """批量处理图像"""
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        # 读取批量图像
        for img_path in batch_paths:
            raw_image = cv2.imread(img_path)
            if raw_image is not None:
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
                batch_images.append(image)
                valid_paths.append(img_path)
            else:
                print(f"Cannot read image: {img_path}")
        
        # 批量处理
        for image, img_path in zip(batch_images, valid_paths):
            features = extractor.extract_frequency_features(image)
            save_frequency_features(img_path, features)

def save_frequency_features(img_path, features):
    """保存频域特征"""
    outdir = os.path.dirname(img_path.replace("image_set", "image_frequency_set"))
    
    # 创建输出目录
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    
    # 保存幅度谱图像
    magnitude_spectrum = features['magnitude_spectrum']
    
    # 归一化到0-255范围
    magnitude_spectrum_normalized = cv2.normalize(
        magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    # 应用颜色映射
    magnitude_spectrum_colored = cv2.applyColorMap(
        magnitude_spectrum_normalized, cv2.COLORMAP_JET
    )
    
    # 保存幅度谱图像
    filename = os.path.basename(img_path)
    output_path = os.path.join(
        outdir, 
        filename[:filename.rfind('.')] + '.png'
    )
    cv2.imwrite(output_path, magnitude_spectrum_colored)

def get_frequency_features(img_path, extractor, DEVICE):
    """提取单张图像的频域特征"""
    # 读取图像
    raw_image = cv2.imread(img_path)
    if raw_image is None:
        print(f"Cannot read image: {img_path}")
        return
    
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    # 提取频域特征
    features = extractor.extract_frequency_features(image)
    
    # 保存特征
    save_frequency_features(img_path, features)

if __name__ == '__main__':
    img_frequency()