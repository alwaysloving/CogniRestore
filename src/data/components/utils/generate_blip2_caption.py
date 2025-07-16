import os, re, glob, argparse, numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm

# 设置环境变量指向镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# -------------------------- CLI --------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--image_root', default='/root/autodl-fs/CognitionCapturer/image_set',
                    help='目录下需有 training_images/ test_images/')
parser.add_argument('--output',      default='/root/autodl-fs/CognitionCapturer/img_description', 
                    help='结果保存子目录')
parser.add_argument('--model',       default='/root/autodl-fs/blip2-model/blip2-model/models--Salesforce--blip2-opt-2.7b/snapshots/59a1ef6c1e5117b3f65523d1c6066825bcf315e3', 
                    help='模型名称或本地路径')
parser.add_argument('--cache_dir',   default='/root/autodl-fs/models/blip2', 
                    help='模型缓存路径')
parser.add_argument('--batch_size',  type=int, default=8)
parser.add_argument('--device',      default='cuda:0')
args = parser.parse_args()

# 确保缓存目录存在
os.makedirs(args.cache_dir, exist_ok=True)

# --------------------- load model once -------------------
device = args.device if torch.cuda.is_available() else 'cpu'
print(f"正在从镜像站下载/加载模型: {args.model}")
print(f"缓存目录: {args.cache_dir}")

# 从镜像站加载模型
processor = Blip2Processor.from_pretrained(
    args.model,
    cache_dir=args.cache_dir
)

model = Blip2ForConditionalGeneration.from_pretrained(
    args.model, 
    cache_dir=args.cache_dir,
    torch_dtype=torch.float16
).to(device).eval()

print(f"模型成功加载到 {device}")

# --------------------- dataset util ----------------------
class ImgDataset(Dataset):
    def __init__(self, files):
        self.files = files
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return img

def get_img_pth(directory):
    def extract_numbers_from_path(path):
        return list(map(int, re.findall(r'\d+', str(path))))
    
    all_paths = glob.glob(os.path.join(str(directory), '**', '*'), recursive=True)
    files = [path for path in all_paths if os.path.isfile(path)]
    imgfiles_sorted = sorted(files, key=extract_numbers_from_path)  # 按数字排序
    
    return imgfiles_sorted

def caption_dir(img_dir: Path):
    files = get_img_pth(img_dir)  # 使用原始代码的图像获取逻辑
    
    # 添加自定义的collate_fn函数
    loader = DataLoader(
        ImgDataset(files), 
        batch_size=args.batch_size, 
        num_workers=4, 
        collate_fn=lambda x: x  # 简单地返回原始批次，不做额外处理
    )
    
    captions = []
    for batch in tqdm(loader, desc=img_dir.name):
        inputs = processor(images=batch, return_tensors='pt').to(device, torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)
        captions += processor.batch_decode(out, skip_special_tokens=True)
    
    return np.array(captions, dtype=object)
# --------------------- run for two splits ----------------
root = Path(args.image_root)
save_dir = Path(args.output)  # 直接使用输出路径，不是子目录
save_dir.mkdir(exist_ok=True)

for split in ['training_images', 'test_images']:
    img_dir = root/split
    if img_dir.exists():
        print(f"正在处理 {split}...")
        caps = caption_dir(img_dir)
        np.save(save_dir/f'texts_BLIP2_{split.split("_")[0]}.npy', caps)
        print(f'已保存 {split}: {len(caps)} 条描述')
    else:
        print(f"警告：目录 {img_dir} 不存在，跳过")