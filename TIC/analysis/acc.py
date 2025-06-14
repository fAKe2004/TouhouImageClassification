import os
import subprocess
import argparse
import gc

import pandas as pd
import torch

import TIC.utils.serve as serve
from TIC.utils.parameter import *

# (model_name, model_type, model_checkpoint)
MODELS = [
    ('ResNet', 'resnet', 'checkpoint/ResNet_model_final.pth'),
    ('ResMoE', 'resmoe', 'checkpoint/ResMoE_epoch7.pth'),
    ('ViT_base', 'vit-base', 'checkpoint/ViT_base_finetune_production_epoch10.pth'),
    ('ViT_large', 'vit-large', 'checkpoint/ViT_large_finetune_production_epoch25.pth'),
    ('ViT_large_filtered', 'vit-large', 'checkpoint/ViT_large_finetune_on_filtered_data_production_epoch5.pth'),
    ('ViT_large_filtered_full_mixed', 'vit-large', 'checkpoint/nViT_epoch17.pth'),
    ('ViT_large_filtered_grey_mixed', 'vit-large', 'checkpoint/nViT_grey_epoch16.pth'),
    ('ViT_large_filtered_grey', 'vit-large', 'checkpoint/nViT_grey_unmix_epoch6.pth'),
    ('ViT_large_full_mixed', 'vit-large', 'checkpoint/nViT_unfiltered_epoch17.pth'),
    ('ViT_large_n', 'vit-large', 'checkpoint/nViT_unfiltered_unaug_epoch3.pth'),
    ('ViT_large_full', 'vit-large', 'checkpoint/nViT_unfiltered_unmix_epoch8.pth'),
]

DEVICE = 'cuda'
RESULT_DIR = 'result'

def get_acc(name):
    df = pd.read_csv(os.path.join(RESULT_DIR, f"{name}.csv"))
    acc = df['correct'].sum() / len(df)
    return {"name" : name, "acc" : acc}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", "-r", action = "store_true")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok = True)
    for name, mtype, checkpoint in MODELS:
        if args.recompute or not os.path.exists(os.path.join(RESULT_DIR, f"{name}.csv")):
            subprocess.run([
                "python", "-m", "TIC.utils.serve", 
                "--model", mtype,
                "--image", TEST_DIR,
                "--device", DEVICE,
                "--weights", checkpoint,
                "--output", os.path.join(RESULT_DIR, f"{name}.csv"),
                "--full",
            ])

    acc = [get_acc(name) for name, _, _ in MODELS]
    df = pd.DataFrame(acc)
    df.to_csv(os.path.join(RESULT_DIR, "acc.csv"), index = False)
