import os
import re
import numpy as np

def natural_key(path: str):
    """按字符串中的数字排序，比如 'img2' < 'img10'"""
    parts = re.split(r'(\d+)', path.lower())
    return [int(p) if p.isdigit() else p for p in parts]

def collect_image_paths(directory: str):
    """收集并排序指定目录下的所有图片路径"""
    exts = {'.jpg', '.jpeg', '.png'}
    paths = []
    for root, _, files in os.walk(directory):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                paths.append(os.path.join(root, fn))
    # 按文件名中数字排序
    return sorted(paths, key=natural_key)

def save_image_paths_to_npy():
    train_dir = "/root/autodl-fs/CognitionCapturer/image_set/training_images"
    test_dir  = "/root/autodl-fs/CognitionCapturer/image_set/test_images"
    output_dir = "/root/autodl-fs/CognitionCapturer/image_set/img_path"
    os.makedirs(output_dir, exist_ok=True)

    train_paths = collect_image_paths(train_dir)
    print(f"找到 {len(train_paths)} 个训练图像")
    np.save(
        os.path.join(output_dir, 'train_image.npy'),
        np.array(train_paths, dtype=object)
    )

    test_paths = collect_image_paths(test_dir)
    print(f"找到 {len(test_paths)} 个测试图像")
    np.save(
        os.path.join(output_dir, 'test_image.npy'),
        np.array(test_paths, dtype=object)
    )

    print(f"文件已保存到 {output_dir}：",
          os.listdir(output_dir))

if __name__ == "__main__":
    save_image_paths_to_npy()
