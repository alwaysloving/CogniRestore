import os
from huggingface_hub import hf_hub_download

# 想把权重放到这个目录下
dest_dir = "/root/autodl-fs/CognitionCapturer/model_pretrained/DepthAnythingWeights"
os.makedirs(dest_dir, exist_ok=True)

# 直接从 HF 下载
weight_path = hf_hub_download(
    repo_id="LiheYoung/depth_anything_vitb14",   # 正确的 repo
    filename="pytorch_model.bin",                 # 仓库里的文件名
    cache_dir=dest_dir                            # 存放路径
)

print("下载完成，文件在：", weight_path)