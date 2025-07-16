"""
这个.py文件用来整ip-adapter的最合适形式with lowest resources
"""
from src.Scripts.train_align.main import Evaler, ExperimentSelecter

"""
reference: https://huggingface.co/docs/diffusers/using-diffusers/controlnet#image-to-image

load multi ip-adapter: https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#multi-ip-adapter
"""
import os
import time

# from src.Visualization.Eval_image.eval_image_metrics import ImgPthGetter, select_pixcorr, process_batches, \
#     EvalerImgMetric

import random
import PIL.Image
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor, CLIPVisionModelWithProjection
from diffusers.utils import load_image
from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler
import matplotlib.pyplot as plt


def show_image(image):
    """
    image: 可以是str， todo或者是pil.image
    """
    if isinstance(image, PIL.Image.Image):
        plt.imshow(image)
        plt.axis('off')  # 不显示坐标轴
        plt.show()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b)


class Generator():
    """
    现在使用了text和depth共同作用
    """

    def __init__(self, sdpath=None, controlnetpath=None, depth_path=None, device=None, modality=None, seed=None,
                 output_path=None, inference_step=None, guidance_scale=None, subject=None, ip_adapter_pth=None,
                 eeg_data_pth=None, eeg_ckpt_pth=None, unet_ckpt=None, image_input=None, debug_mode=None,
                 seperate_generate=False, save_image=True):
        """
        assign 2 params
        path 就是sd和controlnet_depth所对应的储存位置
        """
        ### INITIALIZATION ###
        self.device = device
        self.subject = subject
        self.modality = modality
        self.sdpath = sdpath
        self.controlnet_path = controlnetpath
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(depth_path).to(device)
        self.image_processor = DPTImageProcessor.from_pretrained(depth_path)
        self.ip_adapter_pth = ip_adapter_pth
        self.eeg_data_pth = eeg_data_pth
        self.eeg_ckpt_pth = eeg_ckpt_pth
        self.unet_ckpt = unet_ckpt

        # SD params
        self.inference_step = inference_step
        self.guidance_scale = guidance_scale

        self.image_input = image_input # if not None, use this param as SD's input

        self.debug_mode = debug_mode # if True, only generate 4 EEG embeddings for speed

        self.seperate_generate = seperate_generate # if True, generate image seperately, pipe changed into single IP-Adapter

        self.save_image = save_image # if True, save image in output_path
        self.output_path = output_path
        ### INITIALIZATION ###

        self.seed = seed
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            ip_adapter_pth,
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to(self.device)

        self._load_adapter()

    def _load_adapter(self):
        # reload pipe with controlnet
        self.pipe = AutoPipelineForText2Image.from_pretrained(self.sdpath,
                                                              torch_dtype=torch.float16,
                                                              image_encoder=self.image_encoder,
                                                              safety_checker=None,
                                                              use_safetensors=True,
                                                            #   variant="fp16",
                                                              )
        self.pipe.upcast_vae()
        # load IP-adapter
        if self.seperate_generate is False:
            self.pipe.load_ip_adapter(
                self.ip_adapter_pth,
                subfolder="sdxl_models",
                weight_name=["ip-adapter_sdxl_vit-h.safetensors",
                             "ip-adapter_sdxl_vit-h.safetensors",
                             "ip-adapter_sdxl_vit-h.safetensors"],
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            ### IMPORTANT: YOU CAN CHANGE CONFIGS OF IP-ADAPTERS HERE ! ###
            scales = [
                {
                    "down": {"block_2": [0.0, 1.0]},
                    "up": {"block_0": [0.0, 1.0, 0.0]},
                },
                {
                     "down": {"block_2": [0.0, 0.3]},
                     "up": {"block_0": [0.0, 0.0, 0.0]},
                },
                {
                     "down": {"block_2": [0.0, 0.1]},
                     "up": {"block_0": [0.0, 0.1, 0.0]},
                }
            ]
            self.pipe.set_ip_adapter_scale(scales)
        else:
            # load multi adapter weights
            self.pipe.load_ip_adapter(
                self.ip_adapter_pth,
                subfolder="sdxl_models",
                # "ip-adapter-plus_sd15.safetensors" plus's weight
                weight_name=["ip-adapter_sdxl_vit-h.safetensors"],
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            # set ip_adapter scale (default is 1, which means using only adapter)
            self.pipe.set_ip_adapter_scale(1.0)
        self.pipe.enable_model_cpu_offload(device=self.device)

    def _get_eeg_embeds(self):
        # evaler = Evaler(modality=self.modality, device=self.device, train=False, batch_size=200,
        #                 data_path=self.eeg_data_pth, sub=self.subject, ckpt_path=self.eeg_ckpt_pth,
        #                 unet_ckpt_path=self.unet_ckpt)
        # eeg_embeds = evaler.get_img_features(num_inference_steps=self.inference_step,
        #                                guidance_scale=self.guidance_scale,
        #                                debug_mode=self.debug_mode,
        #                                generator=self.generator)
        # print(f"EEG嵌入结构：{[type(e) for e in eeg_embeds]}")
        # print(f"各部分长度：{[len(e) if hasattr(e, '__len__') else 'scalar' for e in eeg_embeds]}")
        # if len(eeg_embeds) > 0 and hasattr(eeg_embeds[0], 'shape'):
        #     print(f"第一个张量的形状：{eeg_embeds[0].shape}")
        # return eeg_embeds
        evaler = Evaler(modality=self.modality, device=self.device, train=False, batch_size=200,
                    data_path=self.eeg_data_pth, sub=self.subject, ckpt_path=self.eeg_ckpt_pth,
                    unet_ckpt_path=self.unet_ckpt)
        eeg_embeds = evaler.get_img_features(num_inference_steps=self.inference_step,
                                    guidance_scale=self.guidance_scale,
                                    debug_mode=self.debug_mode,
                                    generator=self.generator)
        print(f"EEG嵌入结构：{[type(e) for e in eeg_embeds]}")
        print(f"各部分长度：{[len(e) if hasattr(e, '__len__') else 'scalar' for e in eeg_embeds]}")
        if len(eeg_embeds) > 0 and hasattr(eeg_embeds[0], 'shape'):
            print(f"第一个张量的形状：{eeg_embeds[0].shape}")
        
        # 添加调试代码，检查第7个样本（索引为6）的嵌入
        if len(eeg_embeds) > 0 and isinstance(eeg_embeds[0], list) and len(eeg_embeds[0]) > 0:
            for modal_idx in range(len(eeg_embeds[0])):
                if modal_idx < len(eeg_embeds[0]):
                    embed = eeg_embeds[0][modal_idx]
                    print(f"模态 {modal_idx} 的嵌入形状：{embed.shape if hasattr(embed, 'shape') else '无形状'}")
                    if 6 < embed.shape[0]:  # 检查索引6是否在范围内
                        print(f"样本6，模态{modal_idx}的嵌入大小：{embed[6:7].numel()}")
                        print(f"样本6，模态{modal_idx}的嵌入非零元素数量：{torch.count_nonzero(embed[6:7])}")
                    else:
                        print(f"样本6不在模态{modal_idx}的范围内")
        
        return eeg_embeds

    def _save_image(self, output, image_path):
        image_name = os.path.basename(image_path)
        image_name, image_extension = os.path.splitext(image_name)
        save_pth = f'{self.output_path}'
        save_pth = os.path.join(save_pth, image_name)

        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        if self.seperate_generate is True:
            final = []
            mod_name = ['image', 'text', 'depth']
            for modality_index in range(3):
                ### 0:img 1:text 2:depth 3:final
                final.append(os.path.join(save_pth, f"{mod_name[modality_index]}_{self.seed}{image_extension}"))
                output[modality_index].save(final[modality_index])
        else:
            final = os.path.join(save_pth, f"all_{self.seed}{image_extension}")
            output.save(final)

    def prepare_ip_adapter_embeddings(self, embed):
        """
        给进来(1, 1024)的embedding 返回可输入ipadapter的embedding
        """
        embed = embed.to(torch.float16).to(self.device)
        uncond_image_embeds = torch.zeros_like(embed, dtype=embed.dtype, device=self.device)
        embeds_output = torch.stack([uncond_image_embeds, embed[0:0 + 1]], dim=0)
        #embeds_output = torch.stack([uncond_image_embeds, embed], dim=0)
        return embeds_output

    def __call__(self, gener_mode=-1):  # 在这里传参写想要以什么模式生成image即可？
        """
        genermode:
        0: use image-embedding
        1: use eeg-embedding
        """
        # 1: 写普通的image embedding传入

        if gener_mode == 0:
            image_input = load_image(self.image_input)
            ip_adapter_input = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=[image_input, image_input, image_input],
                ip_adapter_image_embeds=None,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )

        elif gener_mode == 1:
            eeg_embeds = self._get_eeg_embeds()  # eeg_embeds: [gener_embeds, img_embeds, text_embeds, depth_embeds, image_pth]
             # 检查返回的数据结构
            if not eeg_embeds or len(eeg_embeds) < 5:
                print(f"错误：EEG嵌入数据不完整，长度：{len(eeg_embeds) if eeg_embeds else 0}")
                return
            # for i in range(len(eeg_embeds[1])):
            #     img_path = eeg_embeds[4][i]
            #     sim_img = cal_cosine_sim(eeg_embeds[0][0][i:i + 1], eeg_embeds[1][i:i + 1])
            #     sim_text = cal_cosine_sim(eeg_embeds[0][1][i:i + 1], eeg_embeds[2][i:i + 1])
            #     sim_depth = cal_cosine_sim(eeg_embeds[0][2][i:i + 1], eeg_embeds[3][i:i + 1])
            #     image_input = load_image(img_path)
            #     ip_adapter_input = []
                # for j in range(len(eeg_embeds[1])):  # 使用j作为外层循环变量
                #     img_path = eeg_embeds[4][j]
                #     sim_img = cal_cosine_sim(eeg_embeds[0][0][j:j+1], eeg_embeds[1][j:j+1])
                #     sim_text = cal_cosine_sim(eeg_embeds[0][1][j:j+1], eeg_embeds[2][j:j+1])
                #     sim_depth = cal_cosine_sim(eeg_embeds[0][2][j:j+1], eeg_embeds[3][j:j+1])
                #     image_input = load_image(img_path)
                #     ip_adapter_input = []
                #     for i in range(3):  # i表示模态
                #         if eeg_embeds[0][i][j:j+1].numel() > 0:  # 使用j作为样本索引
                #             ip_adapter_input.append(self.prepare_ip_adapter_embeddings(eeg_embeds[0][i][j:j+1]))
                #         else:
                #             print(f"警告：样本{j}的模态{i}的EEG嵌入为空")
                        # 可以提供一个默认的非空嵌入
                # ip_adapter_input = [self.prepare_ip_adapter_embeddings(eeg_embeds[0][0][i:i + 1]),
                #                     self.prepare_ip_adapter_embeddings(eeg_embeds[0][1][i:i + 1]),
                #                     self.prepare_ip_adapter_embeddings(eeg_embeds[0][2][i:i + 1])]
                

        #         if self.seperate_generate is True:
        #             outputs = []
        #             show_image(image_input)
        #             for i in range(len(ip_adapter_input)):
        #                 output = self.pipe(
        #                     prompt="",
        #                     num_inference_steps=5,
        #                     guidance_scale=0.0,  # ?
        #                     negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        #                     generator=self.generator,
        #                     ip_adapter_image_embeds=[ip_adapter_input[i]]
        #                 ).images[0]
        #                 outputs.append(output)
        #                 show_image(output)
        #             if self.save_image is True:
        #                 self._save_image(outputs, img_path)
        #         else:
        #             show_image(image_input)
        #             output = self.pipe(
        #                 prompt="",
        #                 num_inference_steps=5,
        #                 guidance_scale=0.0,  # ?
        #                 negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        #                 generator=self.generator,
        #                 ip_adapter_image_embeds=ip_adapter_input
        #             ).images[0]
        #             show_image(output)
        #             if self.save_image is True:
        #                 self._save_image(output, img_path)
        #             print('hold')
        # else:
        #     raise RuntimeError('Assign genermode pls!')


            
            
            
            total_samples = len(eeg_embeds[1])
            print(f"准备处理 {total_samples} 个样本")
            
            # 使用实际长度而非固定维度
            for sample_idx in range(total_samples):
                try:
                    print(f"处理样本 {sample_idx+1}/{total_samples}")
                    
                    # 检查嵌入是否有效
                    valid_sample = True
                    for modal_idx in range(3):
                        if eeg_embeds[0][modal_idx][sample_idx:sample_idx+1].numel() == 0:
                            print(f"警告：样本 {sample_idx} 模态 {modal_idx} 的嵌入为空，跳过此样本")
                            valid_sample = False
                            break
                    
                    if not valid_sample:
                        continue
                    
                    # 获取对应样本的图片路径
                    img_path = eeg_embeds[4][sample_idx]
                    
                    # 计算相似度 (可选，如果不需要可以注释掉)
                    # sim_img = cal_cosine_sim(eeg_embeds[0][0][sample_idx:sample_idx + 1], eeg_embeds[1][sample_idx:sample_idx + 1])
                    # sim_text = cal_cosine_sim(eeg_embeds[0][1][sample_idx:sample_idx + 1], eeg_embeds[2][sample_idx:sample_idx + 1])
                    # sim_depth = cal_cosine_sim(eeg_embeds[0][2][sample_idx:sample_idx + 1], eeg_embeds[3][sample_idx:sample_idx + 1])
                    
                    # 加载图像并显示
                    image_input = load_image(img_path)
                    show_image(image_input)
                    
                    # 准备IP-Adapter嵌入，并检查每个嵌入是否有效
                    ip_adapter_input = []
                    for modal_idx in range(3):
                        embed = eeg_embeds[0][modal_idx][sample_idx:sample_idx + 1]
                        if embed.numel() > 0:
                            # 检查张量是否包含NaN或Inf值
                            if torch.isnan(embed).any() or torch.isinf(embed).any():
                                print(f"警告：样本 {sample_idx} 模态 {modal_idx} 的嵌入包含NaN或Inf值，跳过此嵌入")
                                continue
                            
                            # 检查张量的形状是否正确 (应该是 [1, 1024] 或类似)
                            if embed.dim() != 2 or embed.shape[1] != 1024:  # 假设正确的形状是 [1, 1024]
                                print(f"警告：样本 {sample_idx} 模态 {modal_idx} 的嵌入形状异常: {embed.shape}，跳过此嵌入")
                                continue
                            
                            processed_embed = self.prepare_ip_adapter_embeddings(embed)
                            ip_adapter_input.append(processed_embed)
                        else:
                            print(f"警告：样本 {sample_idx} 模态 {modal_idx} 的嵌入为空，跳过此嵌入")
                    
                    # 如果没有有效的嵌入，跳过此样本
                    if not ip_adapter_input:
                        print(f"警告：样本 {sample_idx} 没有有效的嵌入，跳过此样本")
                        continue
                    
                    # 生成和保存图像
                    if self.seperate_generate:
                        outputs = []
                        for modality_idx in range(len(ip_adapter_input)):
                            try:
                                output = self.pipe(
                                    prompt="",
                                    num_inference_steps=5,
                                    guidance_scale=0.0,
                                    negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
                                    generator=self.generator,
                                    ip_adapter_image_embeds=[ip_adapter_input[modality_idx]]  # 使用列表包装单个嵌入
                                ).images[0]
                                outputs.append(output)
                                show_image(output)
                            except Exception as e:
                                print(f"生成模态 {modality_idx} 的图像时出错: {str(e)}")
                                # 不添加此模态的输出，继续处理下一个模态
                        
                        # 如果至少生成了一个有效的输出，保存它们
                        if outputs and self.save_image:
                            try:
                                self._save_image(outputs, img_path)
                            except Exception as e:
                                print(f"保存样本 {sample_idx} 的图像时出错: {str(e)}")
                    else:
                        try:
                            output = self.pipe(
                                prompt="",
                                num_inference_steps=5,
                                guidance_scale=0.0,
                                negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
                                generator=self.generator,
                                ip_adapter_image_embeds=ip_adapter_input
                            ).images[0]
                            show_image(output)
                            
                            if self.save_image:
                                self._save_image(output, img_path)
                        except Exception as e:
                            print(f"生成或保存样本 {sample_idx} 的图像时出错: {str(e)}")
                    
                    # 每处理完一个样本后清理内存
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    print(f"处理样本 {sample_idx} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

                    



if __name__ == "__main__":
    # with torch.no_grad():
    #     seeds = [123456, 114514, 567567, 489, 3407]

    #     ### get all subject's ckpt paths ###
    #     base_path = "/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/"
    #     select_param = {"batch_size": 1024}
    #     selecter = ExperimentSelecter(base_path=base_path, select_param=select_param)
    #     eeg_ckpt_path, unet_ckpt_pth = selecter.main(output_unet=True)
    #     ### get all subject's ckpt paths ###

    #     for subject_index in range(10): # 10 subjects
    #         for seed in seeds:
    #             seed_everything(seed)

    #             ### for debug ###
    #             img_pth = "/root/autodl-fs/CognitionCapturer/image_set/test_images/00002_antelope/antelope_01b.jpg"
    #             depth_image_pth = "/root/autodl-fs/CognitionCapturer/image_depth_set/" \
    #                               "test_images/00001_aircraft_carrier/aircraft_carrier_06s.png"
    #             ### for debug ###

    #             ### Config params ###
    #             eeg_data_pth = "/root/autodl-fs/Things_eeg/preprocessed_data_250hz/"
    #             sd_path = "/root/autodl-fs/hub/hub/models--stabilityai--sdxl-turbo/snapshots/71153311d3dbb46851df1931d3ca6e939de83304/"
    #             depth_path = "/root/autodl-fs/dpt-hybrid-midas/dpt-hybrid-midas/models--Intel--dpt-hybrid-midas/snapshots/11eaf7a1cf4bd70740697dbc216f98980c0aeb03/"
    #             control_pth = "/root/autodl-fs/controlnet_sdxl_mid/controlnet_sdxl_mid/models--diffusers--controlnet-depth-sdxl-1.0/snapshots/17bb97973f29801224cd66f192c5ffacf82648b4/"
    #             ip_adapter_pth = "/root/autodl-fs/ip_adapter_models/ip_adapter_models"
    #             subject = f'sub-{subject_index + 1:02d}'
    #             modality = "all"
    #             output_path = f"/data/zkf/ModelWeights/Things_dataset/image_output/{subject}_sdxl"
    #             unet_ckpt = [
    #                 "/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/unet_weights_new/image/model_best.pth",
    #                 "/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/unet_weights_new/text/model_best.pth",
    #                 "/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/unet_weights_new/depth/model_best.pth", ]

    #             debug_mode = True
    #             ### Config params ###

    #             generator = Generator(sdpath=sd_path, controlnetpath=control_pth, depth_path=depth_path,
    #                                   device="cuda:0", modality=modality, seed=seed, output_path=output_path,
    #                                   inference_step=50, guidance_scale=7.5, subject=subject,
    #                                   ip_adapter_pth=ip_adapter_pth, eeg_data_pth=eeg_data_pth,
    #                                   debug_mode=debug_mode, image_input=img_pth, unet_ckpt=unet_ckpt,
    #                                   eeg_ckpt_pth='/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/checkpoints/epoch=012_top200class_accuracy/all=0.47.ckpt', seperate_generate=True,
    #                                   save_image=False)
    #             output = generator(gener_mode=1)
      with torch.no_grad():
        seeds = [123456, 114514, 567567, 489, 3407]
        
        # 手动设置三个模态的UNet路径
        unet_ckpt = [
            "/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/unet_weights_new/image/model_best.pth",
            "/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/unet_weights_new/text/model_best.pth",
            "/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/unet_weights_new/depth/model_best.pth",
        ]
        
        # 跳过ExperimentSelecter，直接指定所有路径
        for seed in seeds:
            seed_everything(seed)

            ### Config params ###
            eeg_data_pth = "/root/autodl-fs/Things_eeg/preprocessed_data_250hz/"
            sd_path = "/root/autodl-fs/hub/hub/models--stabilityai--sdxl-turbo/snapshots/71153311d3dbb46851df1931d3ca6e939de83304/"
            depth_path = "/root/autodl-fs/dpt-hybrid-midas/dpt-hybrid-midas/models--Intel--dpt-hybrid-midas/snapshots/11eaf7a1cf4bd70740697dbc216f98980c0aeb03/"
            control_pth = "/root/autodl-fs/controlnet_sdxl_mid/controlnet_sdxl_mid/models--diffusers--controlnet-depth-sdxl-1.0/snapshots/17bb97973f29801224cd66f192c5ffacf82648b4/"
            ip_adapter_pth = "/root/autodl-fs/ip_adapter_models/ip_adapter_models"
            subject = 'sub-08'  # 直接指定为sub-08
            modality = "all"
            output_path = f"/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/image_output_all/{subject}_sdxl"
            img_pth = "/root/autodl-fs/CognitionCapturer/image_set/test_images"
            debug_mode = False
            ### Config params ###

            # 指定ckpt_path为sub-08的检查点
            eeg_ckpt_path = '/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/checkpoints/epoch=012_top200class_accuracy/all=0.47.ckpt'

            generator = Generator(sdpath=sd_path, 
                                 controlnetpath=control_pth, 
                                 depth_path=depth_path,
                                 device="cuda:0", 
                                 modality=modality, 
                                 seed=seed, 
                                 output_path=output_path,
                                 inference_step=50, 
                                 guidance_scale=7.5, 
                                 subject=subject,
                                 ip_adapter_pth=ip_adapter_pth, 
                                 eeg_data_pth=eeg_data_pth,
                                 debug_mode=debug_mode, 
                                 image_input=img_pth, 
                                 unet_ckpt=unet_ckpt,
                                 eeg_ckpt_pth=eeg_ckpt_path,
                                 seperate_generate=False,
                                 save_image=True)
            
            output = generator(gener_mode=1)
    # with torch.no_grad():
    #     seeds = [123456]  # 只使用一个种子
        
    #     # 手动设置三个模态的UNet路径
    #     unet_ckpt = [
    #         "/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/unet_weights_new/image/model_best.pth",
    #         "/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/unet_weights_new/text/model_best.pth",
    #         "/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/unet_weights_new/depth/model_best.pth",
    #     ]
        
    #     # 只循环一次
    #     for seed in seeds:
    #         seed_everything(seed)

    #         ### Config params ###
    #         eeg_data_pth = "/root/autodl-fs/Things_eeg/preprocessed_data_250hz/"
    #         sd_path = "/root/autodl-fs/hub/hub/models--stabilityai--sdxl-turbo/snapshots/71153311d3dbb46851df1931d3ca6e939de83304/"
    #         depth_path = "/root/autodl-fs/dpt-hybrid-midas/dpt-hybrid-midas/models--Intel--dpt-hybrid-midas/snapshots/11eaf7a1cf4bd70740697dbc216f98980c0aeb03/"
    #         control_pth = "/root/autodl-fs/controlnet_sdxl_mid/controlnet_sdxl_mid/models--diffusers--controlnet-depth-sdxl-1.0/snapshots/17bb97973f29801224cd66f192c5ffacf82648b4/"
    #         ip_adapter_pth = "/root/autodl-fs/ip_adapter_models/ip_adapter_models"
    #         subject = 'sub-08'  # 只指定一个受试者
    #         modality = "all"
    #         output_path = f"/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/image_output/{subject}_sdxl"
    #         img_pth = "/root/autodl-fs/CognitionCapturer/image_set/test_images/"
    #         debug_mode = False
    #         ### Config params ###

    #         # 指定ckpt_path为sub-08的检查点
    #         eeg_ckpt_path = '/root/autodl-fs/CognitionCapturer/ModelWeights/Things_dataset/LightningHydra/logs/train/runs/2025-04-20_16-30-42/checkpoints/epoch=012_top200class_accuracy/all=0.47.ckpt'

    #         generator = Generator(sdpath=sd_path, 
    #                             controlnetpath=control_pth, 
    #                             depth_path=depth_path,
    #                             device="cuda:0", 
    #                             modality=modality, 
    #                             seed=seed, 
    #                             output_path=output_path,
    #                             inference_step=50, 
    #                             guidance_scale=7.5, 
    #                             subject=subject,
    #                             ip_adapter_pth=ip_adapter_pth, 
    #                             eeg_data_pth=eeg_data_pth,
    #                             debug_mode=debug_mode, 
    #                             image_input=img_pth, 
    #                             unet_ckpt=unet_ckpt,
    #                             eeg_ckpt_pth=eeg_ckpt_path,
    #                             seperate_generate=True,
    #                             save_image=True)
            
    #         output = generator(gener_mode=1)
