# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:43:34 2025

@author: nater
"""

import os
import shutil
import random

# Define source and destination paths
source = r""  # Path to main dataset folder 
destination = r""  # Path where Train/Val/Test folders will be created

# Define split ratios
train_ratio = 0.7  
val_ratio = 0.2    
test_ratio = 0.1  

# Create Train, Val and Test folders inside destination
for folder in ["Train", "Val", "Test"]:
    os.makedirs(os.path.join(destination, folder), exist_ok=True)

# Loop through each class folder in the source directory
for item in os.listdir(source):
    class_path = os.path.join(source, item)

    if not os.path.isdir(class_path):
        continue

    # List all images inside the class folder
    images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]
    
    if not images:
        print(f"Warning: No images found in {class_path}")
        continue

    # Shuffle images to ensure random split
    random.shuffle(images)

    # Define number of images for each split
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    
    print(f"Class '{item}': {total_images} images -> {train_count} train, {val_count} validation, {total_images - (train_count + val_count)} test")

    # Create subfolders for each class in Train, Val, and Test
    for folder in ["Train", "Val", "Test"]:
        os.makedirs(os.path.join(destination, folder, item), exist_ok=True)

    # Copy images to respective folders
    for i, img in enumerate(images):
        src_path = os.path.join(class_path, img)
        
        if i < train_count:
            dst_path = os.path.join(destination, "Train", item, img)
        elif i < train_count + val_count:
            dst_path = os.path.join(destination, "Val", item, img)
        else:
            dst_path = os.path.join(destination, "Test", item, img)

        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {img}: {e}")

print("Dataset split completed successfully!")