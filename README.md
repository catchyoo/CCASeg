# CCASeg : Decoding Multi-Scale Context with Convolutional Cross-Attention for Semantic Segmentation (WACV 2025)
Jiwon Yoo<sup>1*</sup>, Dami Ko<sup>1*</sup>, Gyeonghwan Kim<sup>1†</sup>

<sup>*</sup> Equal contribution, <sup>†</sup> Correspondence  
<sup>1</sup> Sogang University

---

This repository contains the official PyTorch implementation of training & evaluation code and the pretrained models for CCASeg.

![CCASeg Architecture](https://github.com/user-attachments/assets/186ae74d-7ab9-4c64-88a1-278bb6fcf1ec)
![CCA Block](https://github.com/user-attachments/assets/5469e50f-1f44-499a-9bd0-a373056de4c8)

---

## Installation

## Evaluation

## Training
Example : Train 'CCASeg-B0' on 'ADE20K'

```bash
# Single-gpu training
CUDA_VISIBLE_DEVICES=0 python ./tools/train.py local_configs/ccaseg/B0/ccaseg.B0.512x512.ade

# Multi-gpu training
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh local_configs/ccaseg/B0/ccaseg.B0.512x512.ade
```

## Citation
