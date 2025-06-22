# Artifact for *TouhouIC* at CS3308-02

---

## To Try Out Our Project:

We host a **web demo** for TouhouIC  at [**touhou.arthas.org**](https://touhou.arthas.org). 

You can upload images, and our model will return predicted character identities.

---

## Repository Structure

This repository contains the artifact for  CS3308-02 course project *TouhouIC: Accurate Image Classifier at Minimal Cost with Transfer Learning and Data Augmentation*.

Repository Structure:
- `checkpoint/`: 
  - model checkpoint folder (excluded in remote repo due to large size)
- `crawler/`: 
  - the Pixiv web crawler to construct raw dataset
- `data/`:
  - dataset folder (excluded in remote repo due to large size)
- `TIC/`: 
  - main python module for TouhouIC
  - `analysis/`: scripts to run evaluation
  - `ResMoE/, ResNet/, TreeViT/, ViT/`: scripts to train models. 
    - data augmentation is implement in `TIC/ViT/ntrain.py`
  - `utils/`: tools to preprocess dataset, filtering , serving etc.
- `reviwer/`：
  - flask web service to manually label clean test dataset
- `web/`:
  - flask web service to serve try-out web demo.
---

## Requirements:
- torch 2.6.0
- lightning 2.5.1

---

## Technical Report:

Available at [doc/report.pdf](doc/report.pdf)

---

## To Reproduce The Main Result:

- Step 1: 
  - Crawl dataset with `crawler/pixiv_crawl.py`; Pixiv Prime account is required to accquire comparatively high-quality data. 
  - Alternatively, you may download raw dataset from [pan.sjtu.edu.cn](https://pan.sjtu.edu.cn/web/share/ffe2bc1ac009a4240ef0c1cb4477da89). Extract to `data/unfiltered`

- Step 2: Set hyperparameters and raw data folder path, finetune **ViT-Base** with
  > python -m TIC.ViT.finetune


- Step 3: 
  - Filter dataset (set weights path in serve.py as needed)
  > python -m TIC.utils.filter --model vit-base

  - Alternatively, download `data_filtered_vit_base.zip` from the above link.

- Step 4: Set hyperparameters and filtered data folder path, finetune **ViT-Large** with
  > python -m TIC.ViT.ntrain.py

- Step 5: Evaluate the model with (adjust weights path as needed)
  > python -m TIC.analysis.acc

- The main result checkpoint (`nViT_epoch17.pth`) and other models' weights are available at [pan.sjtu.edu.cn](
https://pan.sjtu.edu.cn/web/share/1d4d05467a9b0c0b20effdf59bff6fc1). [will be uploaded soon]

---

For other models, run `finetune.py` and `train.py` under respective folder.

For data augmentation ablation study, run `TIC.analysis.aug`.

---

