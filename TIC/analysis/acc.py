import os

import pandas as pd

import TIC.utils.serve as serve
from TIC.utils.parameter import *

# (model_name, model_type, model_checkpoint)
MODELS = [
    ('ResNet', 'resnet', 'checkpoint/ResNet_model_final.pth'),
    ('ResMoE', 'resmoe', 'checkpoint/ResMoE_epoch7.pth'),
    ('ViT_base', 'vit-base', 'checkpoint/ViT_base_finetune_production_epoch10.pth'),
    ('ViT_large', 'vit-large', 'checkpoint/ViT_large_finetune_production_epoch25.pth'),
    ('ViT_large_filtered', 'vit-large', 'checkpoint/ViT_large_finetune_on_filtered_data_production_epoch5.pth'),
    ('nViT', 'vit-large', 'checkpoint/nViT_epoch17.pth'),
    ('nViT_unfiltered', 'vit-large', 'checkpoint/nViT_unfiltered_epoch17.pth'),
    ('nViT_unfiltered_unaug', 'vit-large', 'checkpoint/nViT_unfiltered_unaug_epoch3.pth'),
    ('nViT_unfiltered_unmix', 'vit-large', 'checkpoint/nViT_unfiltered_unmix_epoch8.pth'),
]

DEVICE = 'cuda'
RESULT_DIR = 'result'

if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok = True)
    acc = []
    for name, mtype, checkpoint in MODELS:
        model, transforms, class_to_idx = serve.init(modelt = mtype, weights = checkpoint, device = DEVICE)
        acc.append({"name" : name, "acc" : serve.full_judge(model, transforms, class_to_idx, image = TEST_DIR, device = DEVICE, output = os.path.join(RESULT_DIR, f"{name}.csv"))})

    df = pd.DataFrame(acc)
    df.to_csv(os.path.join(RESULT_DIR, "acc.csv"), index = False)
