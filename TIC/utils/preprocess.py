import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import os
from typing import Tuple, Dict
from tqdm import tqdm # Import tqdm


def get_dataset(data_dir, img_size=(512, 512)) -> Tuple[Dataset]:
    """
    Creates a DataLoader for image classification.

    Args:
        data_dir (str): Path to the root data directory (containing class subfolders).
        batch_size (int): Number of samples per batch.
        img_size (tuple): Target size (height, width) to resize images to.
        shuffle (bool): Whether to shuffle the data each epoch.
        num_workers (int): How many subprocesses to use for data loading.

    Returns:
        tuple: (DataLoader, dict) - The DataLoader instance and a dictionary
               mapping class names to indices.
    """
    # Define standard transformations for image datasets
    # Adjust normalization values (mean, std) if necessary, these are common for ImageNet
    
    try:
        meta_data = torch.load(os.path.join(data_dir, 'meta_mean_std.pth'))
    except FileNotFoundError:
        print("meta_mean_std.pth not found, calcuate mean and std for the first time...")
        mean, std = calculate_mean_std(data_dir, 32, img_size)
        print("calcuate mean and std successfully, saved to meta_mean_std.pth")
    else:
        mean = meta_data['mean']
        std = meta_data['std']
    
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Use ImageFolder - it automatically finds classes based on folder names
    # and assigns indices.
    dataset = ImageFolder(root=data_dir, transform=data_transforms)
    
    return dataset



def calculate_mean_std(data_dir, batch_size, img_size, num_workers=4):
    """
    Calculates the mean and standard deviation of a dataset.

    Args:
        data_dir (str): Path to the root data directory.
        batch_size (int): Batch size for calculation efficiency.
        img_size (tuple): Image size (height, width) to resize to.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: (mean, std) - Tensors containing the per-channel mean and std.
    """
    # Dataset with only Resize and ToTensor
    calc_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor() # Scales images to [0.0, 1.0]
    ])
    dataset = ImageFolder(root=data_dir, transform=calc_transforms)
    # Use a DataLoader to iterate efficiently
    # Shuffle=False is important here, we just need to iterate once
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    std = torch.zeros(3)
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
    torch.save(meta_data, os.path.join(data_dir, 'meta_mean_std.pth'))
    return mean, std

if __name__ == '__main__':
    # Example usage
    DATA_PATH = 'data'
    BATCH_SIZE = 64
    IMG_SIZE = (512, 512)

    # Calculate mean and std
    mean, std = calculate_mean_std(DATA_PATH, BATCH_SIZE, IMG_SIZE)