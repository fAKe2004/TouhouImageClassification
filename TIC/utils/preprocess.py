import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import os
from typing import Tuple, Dict
from tqdm import tqdm # Import tqdm

from TIC.utils.parameter import DATA_DIR, IMAGE_SIZE

META_MEAN_STD_FILENAME = "meta_mean_std.pth"
CLASS_TO_IDX_FILENAME = "class_to_idx.pth"


def get_dataset(data_dir: str = DATA_DIR, image_size = IMAGE_SIZE) -> Dataset:
    """
    Creates a DataLoader for image classification.

    Args:
        data_dir (str): Path to the root data directory (containing class subfolders).
        image_size (tuple): Target size (height, width) to resize images to.

    Returns:
        tuple: (DataLoader, dict) - The DataLoader instance and a dictionary
               mapping class names to indices.
    """
    # Define standard transformations for image datasets
    # Adjust normalization values (mean, std) if necessary, these are common for ImageNet

    data_transforms = get_transforms(data_dir, image_size)

    # Use ImageFolder - it automatically finds classes based on folder names
    # and assigns indices.
    dataset = ImageFolder(root=data_dir, transform=data_transforms)
    
    if not os.path.exists(os.path.join(data_dir, CLASS_TO_IDX_FILENAME)):
        # Save class_to_idx mapping to a file for future reference
        torch.save(dataset.class_to_idx, os.path.join(data_dir, CLASS_TO_IDX_FILENAME))
    
    return dataset

def get_class_to_idx(data_dir: str = DATA_DIR):
    if not os.path.exists(CLASS_TO_IDX_FILENAME):
        _ = get_dataset(data_dir)
    
    return torch.load(os.path.join(data_dir, CLASS_TO_IDX_FILENAME), weights_only=False)

def get_transforms(data_dir: str = DATA_DIR, image_size = IMAGE_SIZE) -> transforms.Compose:
    """
    Creates a transformation pipeline for image classification.

    Args:
        image_size (tuple): Target size (height, width) to resize images to.

    Returns:
        transforms.Compose: The transformation pipeline.
    """
    # Define standard transformations for image datasets
    # Adjust normalization values (mean, std) if necessary, these are common for ImageNet
        
    try:
        meta_data = torch.load(os.path.join(data_dir, META_MEAN_STD_FILENAME), weights_only=False)
    except FileNotFoundError:
        print(f"{META_MEAN_STD_FILENAME} not found, calcuate mean and std for the first time...")
        mean, std = calculate_mean_std(data_dir, 32, image_size)
        print(f"Calcuate mean and std successfully, saved to {META_MEAN_STD_FILENAME}")
    else:
        mean = meta_data['mean']
        std = meta_data['std']
        
    print(f"Normailizing with mean={mean} std={std}")
        
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])



def calculate_mean_std(data_dir, batch_size, image_size, num_workers=4):
    """
    Calculates the mean and standard deviation of a dataset.

    Args:
        data_dir (str): Path to the root data directory.
        batch_size (int): Batch size for calculation efficiency.
        image_size (tuple): Image size (height, width) to resize to.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: (mean, std) - Tensors containing the per-channel mean and std.
    """
    # Dataset with only Resize and ToTensor
    calc_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor() # Scales images to [0.0, 1.0]
    ])
    dataset = ImageFolder(root=data_dir, transform=calc_transforms)
    # Use a DataLoader to iterate efficiently
    # Shuffle=False is important here, we just need to iterate once
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3, dtype=torch.float64) # avoid mean overflow to inf for FP16
    std = torch.zeros(3, dtype=torch.float64)
    n_images = 0

    print(f"Calculating mean and std for dataset at: {data_dir}")
    # Wrap loader with tqdm for progress bar
    for images, _ in tqdm(loader, desc="Calculating Mean/Std"):
        # images shape: [batch_size, channels, height, width]
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        # Reshape images: [batch_size, channels, height * width]
        images = images.view(batch_samples, images.size(1), -1)
        # Calculate mean and std per image in the batch, then mean over the batch
        mean += images.mean([0, 2]) * batch_samples # Sum means weighted by batch size
        std += images.std([0, 2]) * batch_samples   # Sum stds weighted by batch size
        n_images += batch_samples

    mean /= n_images
    std /= n_images

    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std: {std}")
    # Save the calculated mean and std to a file
    meta_data = {'mean': mean, 'std': std}
    torch.save(meta_data, os.path.join(data_dir, META_MEAN_STD_FILENAME))
    return mean, std

if __name__ == '__main__':
    # Example usage
    DATA_PATH = 'data'
    BATCH_SIZE = 32
    image_size = (512, 512)

    # Calculate mean and std
    mean, std = calculate_mean_std(DATA_PATH, BATCH_SIZE, image_size)