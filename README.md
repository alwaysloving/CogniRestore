# CognitionCapturer - Graduation Project

This repository contains the refactored and improved implementation of CognitionCapturer for my graduation project, originally based on the AAAI 2025 paper: "CognitionCapturer: Decoding Visual Stimuli from Human EEG Signals with Multimodal Information."

## Project Overview

This graduation project builds upon the original CognitionCapturer framework, with significant contributions focused on:
- **Code Organization**: Restructured and cleaned up the codebase for better maintainability
- **Logic Streamlining**: Identified and resolved logical inconsistencies in the original implementation
- **Functionality Restoration**: Made the entire pipeline runnable and reproducible
- **Documentation Enhancement**: Improved code documentation and usage instructions

<p align="center">
  <a href="#">
  <p align="center">
    <a href='https://arxiv.org/pdf/2412.10489'><img src='http://img.shields.io/badge/Original%20Paper-arxiv.2412.10489-B31B1B.svg'></a>
  </p>
</p>

<img src="Figs/fig1.png" alt="Framework" style="max-width: 100%; height: auto;"/>
  
<img src="Figs/fig2.png" alt="fig-genexample" style="max-width: 100%; height: auto;"/> 

## My Contributions

### 1. Code Refactoring and Organization
- Reorganized the entire codebase structure for better modularity
- Separated concerns between different components (data processing, model training, evaluation)
- Improved error handling and logging throughout the pipeline

### 2. Logic Streamlining
- Identified and fixed logical inconsistencies in the original training pipeline
- Streamlined the data loading and preprocessing workflows
- Resolved dependency conflicts and version compatibility issues

### 3. Functionality Restoration
- Made the complete training and inference pipeline functional
- Fixed broken scripts and missing dependencies
- Ensured reproducible results across different environments

### 4. Enhanced Documentation
- Added comprehensive inline code documentation
- Improved README with clearer setup and usage instructions
- Created troubleshooting guides for common issues

## Improved Code Structure

```
File path | Description | Improvements Made

/configs
┣ 📂 experiment
┃   ┗ 📜 brainencoder_all.yaml          | ✅ Fixed parameter conflicts and added validation
┃   ┗ 📜 brainencoder_single.yaml       | ✅ Streamlined configuration structure

┣ 📂 paths  
┃   ┗ 📜 pathsSelf.yaml                 | ✅ Added automatic path validation

/src
┣ 📂 data
┃   ┗ 📜 THINGSEEG_datamodule.py        | ✅ Improved data loading efficiency and error handling
┃   ┣ 📂 utils
┃   ┃   ┗ (Processing scripts)          | ✅ Optimized depth map and text generation workflows

┣ 📂 models
┃   ┗ 📜 Cogcap_single_module.py        | ✅ Refactored model architecture for clarity
┃   ┗ 📜 Cogcap_allmodality_module.py   | ✅ Fixed multimodal fusion logic
┃   ┣ 📂 components
┃   ┃   ┗ (Model components)            | ✅ Modularized loss functions and evaluation metrics

┣ 📂 Scripts
┃   ┣ 📂 train_align
┃   ┃   ┗ (Diffusion Prior training)    | ✅ Fixed training instability issues
┃   ┣ 📂 generation
┃   ┃   ┗ (Image generation)            | ✅ Improved generation pipeline reliability
```

## Environment Setup

Install the required dependencies by running:

```bash
conda env create -f environment.yaml
conda activate BCI
```

**Note**: I've updated the environment file to resolve version conflicts and ensure compatibility across different systems.

## Dataset and Model Weights

Dataset follows the Data availability Section of the original [EEG_Image_decode](https://github.com/dongyangli-del/EEG_Image_decode) repository. Please follow their README to download the EEG dataset.

**Reconstruction Results**: Available at https://drive.google.com/file/d/1PfuXZo8JQ7SYsSsYpsUtswdq5gmWKLRU/view?usp=drive_link

**Pre-trained Models Used**:
- SDXL-Turbo: https://huggingface.co/stabilityai/sdxl-turbo
- IP-Adapter: https://huggingface.co/h94/IP-Adapter
- Open CLIP ViT-H/14: https://github.com/mlfoundations/open_clip
- DepthAnything: https://github.com/LiheYoung/Depth-Anything
- BLIP2: https://huggingface.co/docs/transformers/main/model_doc/blip-2

## Usage Instructions

### Setup
1. Download the required datasets and pretrained models
2. Update paths in `configs/paths/pathsSelf.yaml` to match your system
3. Verify environment setup with: `python src/utils/check_setup.py`

### Training Pipeline

#### 1. Classification Training
```bash
python src/train.py experiment=brainencoder_all.yaml
```

#### 2. Diffusion Prior Alignment
```bash
python src/Scripts/train_align/main.py
```

#### 3. Image Generation
```bash
python src/Scripts/generation/multiadapter/ip_adapter_testing.py
```

### Key Improvements in Training
- **Stability**: Fixed training instability issues in the original implementation
- **Efficiency**: Optimized data loading and GPU memory usage
- **Reproducibility**: Added seed management and deterministic training options

## Results and Evaluation

The refactored implementation achieves comparable or improved performance compared to the original work, with enhanced:
- Training stability and convergence
- Code maintainability and readability
- Error handling and debugging capabilities
- Reproducibility across different environments

## Future Work

- [ ] Further optimize memory usage for larger datasets
- [ ] Add support for additional EEG preprocessing techniques
- [ ] Implement more comprehensive evaluation metrics
- [ ] Create a web interface for easy demonstration

## Acknowledgements

### Original Work
This project is based on the outstanding work of:
- Zhang et al. (2025) - CognitionCapturer: Decoding Visual Stimuli from Human EEG Signals with Multimodal Information
- [EEG_Image_decode](https://github.com/dongyangli-del/EEG_Image_decode)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [NICE-EEG](https://github.com/eeyhsong/NICE-EEG)

### Dataset
Original dataset available at: https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub

Special thanks to Dr. Jili Xia for her contributions to the original work.

## Citation

If you use this refactored implementation in your research, please cite both the original paper and acknowledge this graduation project:

```bibtex
@inproceedings{zhang2025cognitioncapturer,
  title={Cognitioncapturer: Decoding visual stimuli from human eeg signal with multimodal information},
  author={Zhang, Kaifan and He, Lihuo and Jiang, Xin and Lu, Wen and Wang, Di and Gao, Xinbo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={13},
  pages={14486--14493},
  year={2025}
}
```