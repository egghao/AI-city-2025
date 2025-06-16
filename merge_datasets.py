import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

def create_directory_structure(base_path):
    """Create the directory structure for the merged dataset."""
    dirs = ['images', 'labels']
    for split in ['train', 'val']:
        for dir_name in dirs:
            os.makedirs(os.path.join(base_path, split, dir_name), exist_ok=True)

def copy_and_rename_files(src_dir, dst_dir, prefix):
    """Copy files from source to destination with a prefix."""
    for file in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(dst_dir, f"{prefix}_{file}")
        shutil.copy2(src_path, dst_path)

def merge_datasets(fisheye_path, visdrone_path, output_path):
    """
    Merge Fisheye8K and synthetic VisDrone datasets.
    
    Args:
        fisheye_path: Path to Fisheye8K dataset
        visdrone_path: Path to synthetic VisDrone dataset
        output_path: Path to save merged dataset
    """
    # Create directory structure
    create_directory_structure(output_path)
    
    # Process training set
    print("Processing training set...")
    for dataset, prefix in [(fisheye_path, 'fisheye'), (visdrone_path, 'visdrone')]:
        # Copy images
        src_img_dir = os.path.join(dataset, 'train', 'images')
        dst_img_dir = os.path.join(output_path, 'train', 'images')
        copy_and_rename_files(src_img_dir, dst_img_dir, prefix)
        
        # Copy labels
        src_label_dir = os.path.join(dataset, 'train', 'labels')
        dst_label_dir = os.path.join(output_path, 'train', 'labels')
        copy_and_rename_files(src_label_dir, dst_label_dir, prefix)
    
    # Process validation set
    print("Processing validation set...")
    for dataset, prefix in [(fisheye_path, 'fisheye'), (visdrone_path, 'visdrone')]:
        # Copy images
        src_img_dir = os.path.join(dataset, 'val', 'images')
        dst_img_dir = os.path.join(output_path, 'val', 'images')
        copy_and_rename_files(src_img_dir, dst_img_dir, prefix)
        
        # Copy labels
        src_label_dir = os.path.join(dataset, 'val', 'labels')
        dst_label_dir = os.path.join(output_path, 'val', 'labels')
        copy_and_rename_files(src_label_dir, dst_label_dir, prefix)

def create_dataset_yaml(output_path):
    """Create YAML configuration file for the merged dataset."""
    config = {
        'path': output_path,
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'bus',
            1: 'bike',
            2: 'car',
            3: 'pedestrian',
            4: 'truck'
        }
    }
    
    with open(os.path.join(output_path, 'merged_dataset.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def main():
    # Define paths
    fisheye_path = 'data/Fisheye8K'
    visdrone_path = 'data/Synthetic_VisDrone'
    output_path = 'data/Merged_Dataset'
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Merge datasets
    print("Starting dataset merge...")
    merge_datasets(fisheye_path, visdrone_path, output_path)
    
    # Create YAML configuration
    print("Creating dataset configuration...")
    create_dataset_yaml(output_path)
    
    print(f"Dataset merge complete. Merged dataset saved to {output_path}")

if __name__ == '__main__':
    main() 