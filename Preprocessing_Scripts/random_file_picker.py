import os
import shutil
import random

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define common image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

# Get all image files in the root folder
image_files = [f for f in os.listdir(script_dir) 
               if os.path.isfile(os.path.join(script_dir, f)) 
               and os.path.splitext(f)[1].lower() in image_extensions]

print(f"Total images found: {len(image_files)}")

# Check if there are at least 50 images
if len(image_files) < 50:
    print(f"Warning: Only {len(image_files)} images found. Selecting all available images.")
    num_to_select = len(image_files)
else:
    num_to_select = 50

# Randomly select images using random.sample() for true randomness
selected_images = random.sample(image_files, num_to_select)

# Create a new directory for the copied images
output_dir = os.path.join(script_dir, "random_images")
os.makedirs(output_dir, exist_ok=True)

# Copy the selected images to the new directory
for image in selected_images:
    src_path = os.path.join(script_dir, image)
    dst_path = os.path.join(output_dir, image)
    try:
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {image}")
    except Exception as e:
        print(f"Error copying {image}: {e}")

print(f"\nTotal images copied: {len(selected_images)}")
print(f"Images saved to: {output_dir}")
