"""
python -m TIC.utils.serve --model vit --image data/伊吹萃香
"""

from TIC.ResNet.model import resnet152
from TIC.ViT.model import ViT
from TIC.utils.parameter import *
from TIC.utils.preprocess import get_transforms, get_class_to_idx

import torch
from PIL import Image
import os
import argparse

model_checkpoints = {
    'resnet': 'checkpoint/ResNet_model_final.pth',
    'vit': 'checkpoint/ViT_model_finetune_base_final.pth'
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
    if model_type.lower() == 'resnet':
        return resnet152(num_classes=num_classes)
    elif model_type.lower() == 'vit':
        return ViT(num_classes=num_classes, pretrained=False)
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
    model = get_model(model_type, num_classes)
    if weights_path is None:
        weights_path = model_checkpoints.get(model_type.lower())
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a model for inference.")
    parser.add_argument('--model', type=str, required=True, choices=['resnet', 'vit'], help='Type of model to load (resnet/vit).')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image or directory of images.')
    parser.add_argument('--weights', type=str, default=None, help='Optional path to model weights file.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu).')
    parser.add_argument('-o', '--output', type=str, default='serve.out', help='Output file to save predictions. (only valid for directory input)')

    args = parser.parse_args()

    print(f"Using device: {args.device}")

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

