'''
this code contains dataset's define and change
todo:IMPORTANT:samples_per_class needs to adjust
'''

import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import open_clip
import os
import json
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer



class FeatureGet:
    def __init__(self):
        self.device = "cuda:0"
        self.vlmodel, self.preprocess_train = self._get_model()

    def _get_model(self):
        vlmodel, preprocess_train, _ = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='/root/autodl-fs/CognitionCapturer/model_pretrained/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_pytorch_mode.bin',
            precision='fp32',
            device=self.device)
        return vlmodel, preprocess_train

    def _Textencoder(self, text):
        text_input = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_feature = self.vlmodel.encode_text(text_input)
            text_feature = F.normalize(text_feature, dim=-1).detach()
        return text_feature


    def _ImageEncoder(self, image):
        img_input = self.preprocess_train(Image.open(image).convert("RGB")).to(self.device).unsqueeze(0)
        with torch.no_grad():
            image_feature = self.vlmodel.encode_image(img_input)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
        return image_feature

    def get_text_feature(self, type):
        saved_text_path = "/root/autodl-fs/CognitionCapturer/image_set/img_description"
        saved_img_path = "/root/autodl-fs/CognitionCapturer/image_set/img_path/"
        with torch.no_grad():
            # 'train.npy' , 'test.npy'
            for text_file_name, img_file_name in zip(['texts_BLIP2_train.npy'], ['train_image.npy']):
                text_features_dict = {}
                textfile_name = os.path.join(saved_text_path, text_file_name)
                img_file_name = os.path.join(saved_img_path, img_file_name)
                text_path = np.load(textfile_name,allow_pickle=True).tolist()
                img_path = np.load(img_file_name,allow_pickle=True).tolist()
                for j, text_description in enumerate(text_path):

                    print(f"{j + 1} / {len(text_path)}")

                    if type != 'finegrain':
                        raise NotImplementedError("todo")

                    ### model forward ###
                    output = self._Textencoder(text_description)

                    ### to cpu and append to list, then empty cache ###
                    output = output.cpu()
                    text_features_dict[os.path.basename(img_path[j])] = output
                    torch.cuda.empty_cache()
                
                # torch.save(text_features_dict,
                #            f"/root/autodl-fs/CognitionCapturer/model_pretrained/data_features/text_{type}_features_clip_dict.pt")
                out_dir = "/root/autodl-fs/CognitionCapturer/model_pretrained/data_features"
                os.makedirs(out_dir, exist_ok=True)
                torch.save(
                    text_features_dict,
                    os.path.join(out_dir, f"text_{type}_features_clip_dict.pt")
                )
                
    def get_feature(self, type):
        saved_path = "/root/autodl-fs/CognitionCapturer/image_set/img_path/"
        print(f"Starting get_feature with type: {type}")
        
        with torch.no_grad():
            for i, file_name in enumerate(['train_image.npy']):
                image_features_dict = {}
                file_path = os.path.join(saved_path, file_name)
                print(f"Loading file: {file_path}")
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"ERROR: File does not exist: {file_path}")
                    continue
                
                path = np.load(file_path, allow_pickle=True).tolist()
                print(f"Loaded {len(path)} image paths")
                
                success_count = 0
                error_count = 0
                
                for j, img_path in enumerate(path):
                    print(f"{j + 1} / {len(path)} - Processing: {img_path}")
                    
                    # 保存原始文件名作为字典的键
                    original_filename = os.path.basename(img_path)
                    
                    if type == 'depth':
                        img_path = img_path.replace("image_set", "image_depth_set").replace(".jpg", ".png")
                    if type == 'frequency':
                        img_path = img_path.replace("image_set", "image_frequency_set")
                        img_path = img_path.replace(".jpg", ".png")
                    if type == 'aug':
                        img_path = img_path.replace("training_images", "aug_images")
                    
                    # 检查转换后的文件是否存在
                    if not os.path.exists(img_path):
                        print(f"ERROR: Converted file does not exist: {img_path}")
                        error_count += 1
                        continue
                    
                    try:
                        # model forward
                        output = self._ImageEncoder(img_path)
                        
                        # to cpu and append to list, then empty cache
                        output = output.cpu()
                        
                        # 使用原始文件名作为键，确保所有特征字典使用相同的键
                        image_features_dict[original_filename] = output
                        torch.cuda.empty_cache()
                        success_count += 1
                        
                    except Exception as e:
                        print(f"ERROR processing {img_path}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        error_count += 1
                
                print(f"Processing complete. Success: {success_count}, Errors: {error_count}")
                print(f"Dictionary size: {len(image_features_dict)}")
                
                # 保存路径
                save_path = f"/root/autodl-fs/CognitionCapturer/model_pretrained/data_features/image_{type}_features_clip_dict.pt"
                print(f"Attempting to save to: {save_path}")
                
                # 检查保存目录是否存在
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    print(f"Creating directory: {save_dir}")
                    os.makedirs(save_dir, exist_ok=True)
                
                # 尝试保存
                try:
                    torch.save(image_features_dict, save_path)
                    print(f"Successfully saved to: {save_path}")
                    
                    # 验证文件是否已保存
                    if os.path.exists(save_path):
                        file_size = os.path.getsize(save_path)
                        print(f"File exists. Size: {file_size} bytes")
                    else:
                        print(f"ERROR: File was not saved!")
                        
                except Exception as e:
                    print(f"ERROR saving file: {str(e)}")
                    import traceback
                    traceback.print_exc()


def open_img(img_path):
    '''
    input : img path
    output : img with plt showed
    '''
    img = Image.open(img_path)
    plt.figure(f"{img_path}")
    plt.imshow(img)
    plt.axis('on')
    plt.title('image')
    plt.show()


if __name__ == "__main__":
    try:
        print("Starting feature extraction...")
        instance = FeatureGet()
        print("Model loaded successfully")
        
        instance.get_feature("frequency")
        print("Feature extraction completed")
        
    except Exception as e:
        print(f"ERROR in main: {str(e)}")
        import traceback
        traceback.print_exc()