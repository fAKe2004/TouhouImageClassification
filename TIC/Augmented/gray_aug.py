import torch
import torchvision.transforms as T
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image # Import for saving tensors as images
import os
from PIL import Image # PIL is used by torchvision for some operations and can be a fallback

def load_and_save_images_as_grayscale(root_dir, output_root_dir = None):
    """
    Loads all images from a directory and its subdirectories,
    converts them to grayscale, and saves them to a new directory
    while preserving the original subdirectory structure.

    Args:
        root_dir (str): The path to the root directory containing original images.
        output_root_dir (str): The path to the root directory where grayscale images will be saved.

    Returns:
        int: The number of images successfully processed and saved.
    """
    processed_count = 0
    # Define the transformation to grayscale
    grayscale_transform = T.Grayscale(num_output_channels=1)
    
    if output_root_dir is None:
        output_root_dir = root_dir

    # Supported image extensions
    supported_extensions = ('.png', '.jpg', '.jpeg')

    for subdir, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(supported_extensions):
                img_path = os.path.join(subdir, file_name)
                
                # Determine the relative path to maintain directory structure
                relative_path = os.path.relpath(subdir, root_dir)
                output_subdir = os.path.join(output_root_dir, relative_path)
                
                # Ensure the output subdirectory exists
                os.makedirs(output_subdir, exist_ok=True)
                
                # Define the output path for the grayscale image
                base, ext = os.path.splitext(file_name)
                output_img_path = os.path.join(output_subdir, f"{base}_gray{ext}") # Append _gray to filename

                try:
                    img = read_image(img_path)

                    if img.shape[0] == 4:
                        img = T.functional.rgba_to_rgb(img)
                    
                    gray_img = grayscale_transform(img)
                    gray_img = gray_img / 255.0
                    gray_img = gray_img.clamp(0, 1) 
                    # print(f"Image after grayscale transform: {gray_img}")
                    save_image(gray_img, output_img_path)
                    print(f"Loaded, converted, and saved {img_path} to {output_img_path}. Shape: {gray_img.shape}")
                    processed_count += 1
                except Exception as e:
                    print(f"Could not read or convert image {img_path}: {e}")
                    exit()
                    try:
                        pil_img = Image.open(img_path)
                        if pil_img.mode == 'P' or pil_img.mode == 'RGBA' or pil_img.mode == 'LA':
                            pil_img = pil_img.convert('RGB')

                        tensor_img = T.ToTensor()(pil_img) # Converts to [0, 1] float tensor
                        gray_img = grayscale_transform(tensor_img)
                        
                        save_image(gray_img, output_img_path) # save_image expects [0,1] for float
                        print(f"Fallback: Loaded, converted, and saved {img_path} to {output_img_path} using PIL. Shape: {gray_img.shape}")
                        processed_count += 1
                    except Exception as e_pil:
                        print(f"PIL fallback also failed for {img_path}: {e_pil}")

    return processed_count

if __name__ == "__main__":
    image_directory = '../../crawler/data_filtered_vit_base'
    output_grayscale_directory = image_directory

    if not os.path.isdir(image_directory):
        print(f"Error: Input directory '{image_directory}' not found.")
    else:
        # Create the main output directory if it doesn't exist
        os.makedirs(output_grayscale_directory, exist_ok=True)
        print(f"Loading images from: {image_directory}")
        print(f"Saving grayscale images to: {output_grayscale_directory}")
        
        num_saved_images = load_and_save_images_as_grayscale(image_directory, output_grayscale_directory)
        
        print(f"\nSuccessfully processed and saved {num_saved_images} images to {output_grayscale_directory}.")
