"""
python -m TIC.utils.calc_tfpn --model vit-base --image_dir test
"""


from ast import Assert
from cProfile import label
from TIC.ResMoE.model import MoEClassifier
from TIC.ResNet.model import resnet152
from TIC.ViT.model import ViT
import TIC.ResMoE.train as moet
from TIC.utils.parameter import *
from TIC.utils.preprocess import get_transforms, get_class_to_idx
from TIC.utils.serve import serve, init

import torch
from PIL import Image
import os
import argparse
import tqdm


def calc_tfpn(model, transforms, class_to_idx, args = None, image_dir = None, device = None):
    '''
    returns TP/TN/FP/FN
    '''
    
    if args:
        image_dir = args.image_dir
        device = args.device
    
    tot = 0
    
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if (os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.invalid']):
                tot += 1
    print(f"Total images to process: {tot}")
    bar = tqdm.tqdm(total=tot, desc="Processing images", unit="image")
    
    class_metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    
    for root, dirs, files in os.walk(image_dir): 
        for filename in files:
            file_path = os.path.join(root, filename)
            label = os.path.basename(root)
            is_valid = 1 if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png'] else 0
            
            print(f'--- Processing {filename} ---')
            try:
                image = Image.open(file_path).convert('RGB')
                image_tensor = transforms(image).unsqueeze(0)
                predict_class, confidence = serve(model, image_tensor, class_to_idx, device)
                
                print(f'Predicted class: {predict_class}, Confidence: {confidence} Correct: {predict_class == label}')
                
                pred_success = 1 if predict_class == label else 0
                
                class_metrics['TP'] += pred_success * is_valid
                class_metrics['TN'] += (1 - pred_success) * (1 - is_valid)
                class_metrics['FP'] += pred_success * (1 - is_valid)
                class_metrics['FN'] += (1 - pred_success) * is_valid
            except Exception as e:
                print(f"Error opening image {filename}: {e}")
                
            bar.update(1)
            
    bar.close()
    print(f"TP: {class_metrics['TP']}, TN: {class_metrics['TN']}, FP: {class_metrics['FP']}, FN: {class_metrics['FN']}, TOTAL images: {tot}")
    
    Assert(tot == class_metrics['TP'] + class_metrics['TN'] + class_metrics['FP'] + class_metrics['FN'], 
           msg="Total images processed does not match total images counted")
    
    print(f"Accuracy: {((class_metrics['TP'] + class_metrics['TN']) / tot) * 100:.2f}%")
    print(f"Precision: {class_metrics['TP']/(class_metrics['TP'] + class_metrics['FP'])}")
    print(f"Recall: {class_metrics['TP'] / (class_metrics['TP'] + class_metrics['FN'])}")
    print(f"F1 Score: {2 * class_metrics['TP'] / (2 * class_metrics['TP'] + class_metrics['FP'] + class_metrics['FN'])}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate TP, TN, FP, FN for a model on a dataset")
    parser.add_argument('--model', type=str, required=True, choices=['resnet', 'vit-base', 'vit-large', 'nvit', 'resmoe'], help='Type of model to load (resnet/vit/nvit/resmoe).')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the input image or directory of images.')
    parser.add_argument('--weights', type=str, default=None, help='Optional path to model weights file.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu).')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    model, transforms, class_to_idx = init(args)
    
    print(f"--- Calculating TP, TN, FP, FN for {args.model} on {args.image_dir} ---")
    
    calc_tfpn(model, transforms, class_to_idx, args=args)
    
    print(f"--- Finished calculating TP, TN, FP, FN for {args.model} on {args.image_dir} ---")
    
    
    