# CCASeg : Decoding Multi-Scale Context with Convolutional Cross-Attention for Semantic Segmentation (WACV 2025)

> #### Jiwon Yoo<sup>1</sup>\*, Dami Ko<sup>1</sup>\*, Gyeonghwan Kim<sup>1&dagger;</sup>
> \* Equal contribution, <sup>&dagger;</sup>Correspondence

> <sup>1</sup> Sogang University

---

This repository contains the official PyTorch implementation of training & evaluation code and the pretrained models for CCASeg.

![CCASeg Architecture](https://github.com/user-attachments/assets/186ae74d-7ab9-4c64-88a1-278bb6fcf1ec)
![CCA Block](https://github.com/user-attachments/assets/5469e50f-1f44-499a-9bd0-a373056de4c8)

---

## Installation
For install and data preparation, please refer to the guidelines in [MMSegmentation v1.0.0](https://github.com/open-mmlab/mmsegmentation?tab=readme-ov-file)
Other requirements: pip install timm==0.9.12
An example (works for me): CUDA 11.8 and pytorch 2.0.1

```bash
pip install torchvision==0.16.0
pip install timm==0.9.12
pip install mmcv==2.1.0
pip install opencv-python==4.8.1.78
cd CCASeg && pip install -e . --user
```

## Evaluation
Download [CCASeg weights](https://drive.google.com/drive/folders/1hKgzJ0vhjhPcG5TRG0dkA5YpD8N8qyPp?hl=ko) into the ``/path/to/checkpoint_file``.

``local_configs/`` contains config files. 

Example : Evaluate ``CCASeg-B0`` on ``ADE20K``:

```bash
# Single-gpu testing
CUDA_VISIBLE_DEVICES=0 python ./tools/test.py local_configs/ccaseg/B0/ccaseg.b0.512x512.ade.160k.py /path/to/checkpoint_file

# Multi-gpu testing
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_test.sh local_configs/ccaseg/B0/ccaseg.b0.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM>
```
## Training
Download backbone(MiT-B0 & MiT-B1 & MSCAN-T & MSCAN-B) pretrained weights in [here](https://drive.google.com/drive/folders/1Wr4qiaH54IywMEIJ39w-5X3-MKVOxYi1?hl=ko).

Put them in a folder ``ckpt/``.

Example : Train ``CCASeg-B0`` on ``ADE20K``:

```bash
# Single-gpu training
CUDA_VISIBLE_DEVICES=0 python ./tools/train.py local_configs/ccaseg/B0/ccaseg.b0.512x512.ade.160k.py 

# Multi-gpu training
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh local_configs/ccaseg/B0/ccaseg.b0.512x512.ade.160k.py <GPU_NUM>
```

## Citation
```bash
@InProceedings{Yoo_2025_WACV,
    author    = {Yoo, Jiwon and Ko, Dami and Kim, Gyeonghwan},
    title     = {CCASeg: Decoding Multi-Scale Context with Convolutional Cross-Attention for Semantic Segmentation},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {9461-9470}
}
```
