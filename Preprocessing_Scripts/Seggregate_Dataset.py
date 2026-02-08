import os
import shutil
import random

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define common image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

# Create train, val, and test directories
train_dir = os.path.join(script_dir, "train")
val_dir = os.path.join(script_dir, "val")
test_dir = os.path.join(script_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all subdirectories in the root folder
subdirs = [d for d in os.listdir(script_dir) 
           if os.path.isdir(os.path.join(script_dir, d)) 
           and d not in ['train', 'val', 'test']]

print(f"Found {len(subdirs)} folders to process")
print("Split ratio: 70% train, 15% val, 15% test\n")

# Process each subdirectory
for folder_name in subdirs:
    folder_path = os.path.join(script_dir, folder_name)
    
    # Get all image files in this folder
    image_files = [f for f in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, f))
                   and os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {folder_name}, skipping...")
        continue
    
    print(f"Processing {folder_name}: {len(image_files)} images found")
    
    # Randomly shuffle and split images 70-15-15
    random.shuffle(image_files)
    train_split = int(len(image_files) * 0.7)
    val_split = int(len(image_files) * 0.85)
    
    train_images = image_files[:train_split]
    val_images = image_files[train_split:val_split]
    test_images = image_files[val_split:]
    
    # Create subdirectories in train, val, and test
    train_subdir = os.path.join(train_dir, folder_name)
    val_subdir = os.path.join(val_dir, folder_name)
    test_subdir = os.path.join(test_dir, folder_name)
    
    os.makedirs(train_subdir, exist_ok=True)
    os.makedirs(val_subdir, exist_ok=True)
    os.makedirs(test_subdir, exist_ok=True)
    
    # Copy images to train directory (70%)
    for image in train_images:
        src_path = os.path.join(folder_path, image)
        dst_path = os.path.join(train_subdir, image)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {image} to train: {e}")
    
    # Copy images to val directory (15%)
    for image in val_images:
        src_path = os.path.join(folder_path, image)
        dst_path = os.path.join(val_subdir, image)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {image} to val: {e}")
    
    # Copy images to test directory (15%)
    for image in test_images:
        src_path = os.path.join(folder_path, image)
        dst_path = os.path.join(test_subdir, image)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {image} to test: {e}")
    
    print(f"  Train: {len(train_images)} images | Val: {len(val_images)} images | Test: {len(test_images)} images")

print(f"\n✓ All folders split successfully!")
print(f"Train directory: {train_dir}")
print(f"Val directory: {val_dir}")
print(f"Test directory: {test_dir}")
