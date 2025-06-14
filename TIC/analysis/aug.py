import os
import argparse
from PIL import Image

import torchvision as tv
import torchvision.transforms.v2 as v2

from TIC.utils.parameter import VIT_IMAGE_SIZE

# (name, numbers, transforms)

AUGS = [
    ("original", 1, v2.Resize(VIT_IMAGE_SIZE)),
    ("only_grey", 1, v2.Compose([
        v2.Resize(VIT_IMAGE_SIZE),
        v2.RandomGrayscale(p=1.0),
    ])),
    ("only_colorjitter", 8, v2.Compose([
        v2.Resize(VIT_IMAGE_SIZE),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])),
    ("full", 16, v2.Compose([
        v2.RandomResizedCrop(VIT_IMAGE_SIZE),          
        v2.RandomHorizontalFlip(),         
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        v2.RandomGrayscale(p=0.2),
        v2.RandomErasing(p=0.5),
    ])),
]

STORE_DIR = "show_augmentation"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", "-t", type = str, default = None)
    args = parser.parse_args()

    img = Image.open(args.target)

    os.makedirs(STORE_DIR, exist_ok = True)

    for name, numbers, transforms in AUGS:
        for i in range(numbers):
            aug = transforms(img)
            aug.save(os.path.join(STORE_DIR, f"{name}_{i}.png"))
