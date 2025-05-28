"""
python -m TIC.utils.serve --model vit --image data/伊吹萃香
"""

from TIC.ResNet.model import resnet152
from TIC.ViT.model import ViT
from TIC.ViT.ntrain import ViTLModule
from TIC.ResMoE.train import ResMoETrainerModule
from TIC.utils.parameter import *
from TIC.utils.preprocess import get_transforms, get_class_to_idx

import torch
from PIL import Image
import os
import argparse
import tqdm

model_checkpoints = {
    'resnet': 'checkpoint/ResNet_model_final.pth',
    'vit-base': 'checkpoint/ViT_base_finetune_production_epoch10.pth',
    'vit-large': 'checkpoint/ViT_large_finetune_production_epoch25.pth',
    'nvit': 'checkpoint/nViT.ckpt',
    'resmoe': 'checkpoint/ResMoE.ckpt'
}

def get_model(model_type: str, num_classes: int):
    """
    Returns a model instance based on the specified type.

    Args:
        model_type (str): The type of model to load. Options are 'resnet', or 'vit'.
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The model instance.
    """
    model_type = model_type.lower().replace('_', '-')
    if model_type == 'resnet':
        return resnet152(num_classes=num_classes)
    elif model_type == 'vit-base':
        return ViT(num_classes=num_classes, pretrained=False, model_name='google/vit-base-patch16-224-in21k')
    elif model_type == 'vit-large':
        return ViT(num_classes=num_classes, pretrained=False, model_name='google/vit-large-patch16-224-in21k')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_model(model_type: str, num_classes: int, weights_path: str = None, device: str = 'cuda'):
    """
    Loads the model structure and weights.

    Args:
        model_type (str): The type of model ('resnet' or 'vit').
        num_classes (int): Number of classes the model was trained on.
        weights_path (str, optional): Path to the model weights file. If None, uses default.
        device (str): Device to load the model onto ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The loaded model instance.
    """
    if model_type == 'nvit':
        model = ViTLModule.load_from_checkpoint(weights_path).vit
    elif model_type == 'resmoe':
        model = ResMoETrainerModule.load_from_checkpoint(weights_path).model
    else:
        model_type = model_type.lower().replace('_', '-')
        model = get_model(model_type, num_classes)
        
        if weights_path is None:
            weights_path = model_checkpoints.get(model_type)
            if weights_path is None:
                raise ValueError(f"No default checkpoint found for model type: {model_type}")
            print(f"Loading default weights from: {weights_path}")
        else:
            print(f"Loading weights from specified path: {weights_path}")

        ckpt = torch.load(weights_path, map_location=torch.device(device), weights_only=False)
        if isinstance(ckpt, tuple):
            model_state_dict = ckpt[0]
        else:
            model_state_dict = ckpt

        model.load_state_dict(model_state_dict)

    model.to(device)
    print(f"Model loaded successfully onto {device}.")
    return model

def serve(model, image_tensor, class_to_idx, device: str = 'cuda'):
    """
    Serve the model for inference on a single preprocessed image tensor.

    Args:
        model (torch.nn.Module): The loaded model instance.
        image_tensor (torch.Tensor): The preprocessed input image tensor (batch dimension added).
        class_to_idx (dict): Mapping from class names to indices.
        device (str): Device the model and tensor are on ('cuda' or 'cpu').

    Returns:
        str: The predicted class name.
    """
    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)

        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = idx_to_class[predicted_idx.item()]

    return predicted_class, confidence.item()

def init(args):
    '''
    Model initialization function.
    Parameters:
        args: Command line arguments containing model type, weights path, and device.
    Returns:
        model: The loaded model instance.
        transforms: The image transformation pipeline.
        class_to_idx: Mapping from class names to indices.
    '''
    print("Loading class mapping...")
    try:
        class_to_idx = get_class_to_idx(DATA_DIR)
        num_classes = len(class_to_idx)
        print(f"Loaded {num_classes} classes.")
    except Exception as e:
        print(f"Error loading class mapping: {e}")
        exit(1)

    print(f"Loading model '{args.model}'...")
    try:
        model = load_model(args.model, num_classes, args.weights, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    print("Getting image transformations...")
    current_image_size = get_image_size(args.model)
    try:
        transforms = get_transforms(DATA_DIR, current_image_size)
        print(f"Using image size: {current_image_size}")
    except Exception as e:
        print(f"Error getting transformations: {e}")
        exit(1)
    
    return model, transforms, class_to_idx

def full_judge(model, transforms, class_to_idx, args):
    '''
    Full judgement function.
    Walk through the directory, predict for every image, and save the results if output is given.
    '''
    # Check if the input is a file or directory
    if os.path.isfile(args.image):
        print(f"Processing single image: {args.image}")
        try:
            image = Image.open(args.image).convert('RGB')
            image_tensor = transforms(image).unsqueeze(0)
            predicted_class, confidence = serve(model, image_tensor, class_to_idx, args.device)
            print(f"Prediction: {predicted_class} (Confidence: {confidence:.4f})")
        except Exception as e:
            print(f"Error processing image {args.image}: {e}")
        return

    # Generate the output file's header
    if (args.output):
        with open(args.output, 'w') as f:
            print(f"filename,predicted_class,confidence,actual_class,correct,path", file=f)

    tot = 0
    cnt = 0
    correct_cnt = 0

    # Count the number of images
    for root, dirs, files in os.walk(args.image):
        for filename in files:
            if (os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
                tot += 1
    print(f"Total images to process: {tot}")
    bar = tqdm.tqdm(total=tot, desc="Processing images", unit="image")

    # Walk through the directory
    for root, dirs, files in os.walk(args.image):
        for filename in files:
            file_path = os.path.join(root, filename)
            label = os.path.basename(root)
            if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                print(f"--- Processing: {filename} ---")
                try:
                    image = Image.open(file_path).convert('RGB')
                    image_tensor = transforms(image).unsqueeze(0)
                    predicted_class, confidence = serve(model, image_tensor, class_to_idx, args.device)
                    print(f"Prediction: {predicted_class} (Confidence: {confidence:.4f}) Correct: {predicted_class == label}")
                    cnt += 1
                    correct_cnt += (predicted_class == label)
                    
                    if args.output:
                        with open(args.output, 'a') as f:
                            f.write(f"{filename},{predicted_class},{confidence:.4f},{label},{predicted_class == label},{file_path}\n")
                
                except Exception as e:
                    print(f"Error processing image {filename}: {e}")
                bar.update(1)
            else:
                print(f"Skipping non-image file: {filename}")
    print(f"Total images processed: {cnt}, Correct predictions: {correct_cnt}, Accuracy: {correct_cnt / cnt * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a model for inference.")
    parser.add_argument('--model', type=str, required=True, choices=['resnet', 'vit-base', 'vit-large'], help='Type of model to load (resnet/vit).')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image or directory of images.')
    parser.add_argument('--weights', type=str, default=None, help='Optional path to model weights file.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu).')
    parser.add_argument('-o', '--output', type=str, default='serve.out', help='Output file to save predictions. (only valid for directory input)')
    parser.add_argument('--full', action='store_true', help='Full judgement mode, process all images in the directory.')

    args = parser.parse_args()

    print(f"Using device: {args.device}")
    model, transforms, class_to_idx = init(args)

    if (args.full):
        print("Full judgement mode activated.")
        full_judge(model, transforms, class_to_idx, args)
        exit(0)

    if os.path.isfile(args.image):
        print(f"Processing single image: {args.image}")
        try:
            image = Image.open(args.image).convert('RGB')
            image_tensor = transforms(image).unsqueeze(0)
            predicted_class, confidence = serve(model, image_tensor, class_to_idx, args.device)
            print(f"Prediction: {predicted_class} (Confidence: {confidence:.4f})")
        except Exception as e:
            print(f"Error processing image {args.image}: {e}")

    elif os.path.isdir(args.image):
        print(f"Processing images in directory: {args.image}")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        with open(args.output, 'a') as f:
          for filename in os.listdir(args.image):
              file_path = os.path.join(args.image, filename)
              if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in image_extensions:
                  print(f"--- Processing: {filename} ---")
                  try:
                      image = Image.open(file_path).convert('RGB')
                      image_tensor = transforms(image).unsqueeze(0)
                      predicted_class, confidence = serve(model, image_tensor, class_to_idx, args.device)
                      print(f"Prediction: {predicted_class} (Confidence: {confidence:.4f})")
                      
                      f.write(f"{filename} {predicted_class} {confidence:.4f}\n")
                  
                  except Exception as e:
                      print(f"Error processing image {filename}: {e}")
              else:
                  print(f"Skipping non-image file or sub-directory: {filename}")
    else:
        print(f"Error: Image path '{args.image}' is not a valid file or directory.")
        exit(1)

    print("Inference complete.")

