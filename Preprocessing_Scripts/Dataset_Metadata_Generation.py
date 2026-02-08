import os
import json
from pathlib import Path
from datetime import datetime

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define common image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

# Initialize metadata structure
metadata = {
    "generated_at": datetime.now().isoformat(),
    "dataset_root": script_dir,
    "splits": {},
    "classes": {},
    "statistics": {
        "total_images": 0,
        "total_size_mb": 0,
        "image_formats": {}
    }
}

# Define splits to analyze
splits = ['train', 'val', 'test']
total_images = 0
total_size = 0

print("Generating dataset metadata...\n")

# Process each split
for split_name in splits:
    split_path = os.path.join(script_dir, split_name)
    
    if not os.path.isdir(split_path):
        print(f"⚠ {split_name}/ folder not found, skipping...")
        continue
    
    metadata["splits"][split_name] = {
        "path": split_name,
        "total_images": 0,
        "categories": {}
    }
    
    # Get all subdirectories (classes/categories)
    classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    
    for class_name in classes:
        class_path = os.path.join(split_path, class_name)
        
        # Get all image files
        image_files = [f for f in os.listdir(class_path)
                       if os.path.isfile(os.path.join(class_path, f))
                       and os.path.splitext(f)[1].lower() in image_extensions]
        
        if not image_files:
            continue
        
        # Calculate size and format info
        class_size = 0
        format_count = {}
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            class_size += os.path.getsize(img_path)
            
            ext = os.path.splitext(img_file)[1].lower()
            format_count[ext] = format_count.get(ext, 0) + 1
        
        # Add to metadata
        metadata["splits"][split_name]["categories"][class_name] = {
            "count": len(image_files),
            "size_mb": round(class_size / (1024 * 1024), 2),
            "formats": format_count
        }
        
        metadata["splits"][split_name]["total_images"] += len(image_files)
        total_images += len(image_files)
        total_size += class_size
        
        # Update image format statistics
        for fmt, count in format_count.items():
            metadata["statistics"]["image_formats"][fmt] = metadata["statistics"]["image_formats"].get(fmt, 0) + count
        
        # Update class statistics
        if class_name not in metadata["classes"]:
            metadata["classes"][class_name] = 0
        metadata["classes"][class_name] += len(image_files)
    
    print(f"✓ {split_name}: {metadata['splits'][split_name]['total_images']} images from {len(classes)} classes")

# Update total statistics
metadata["statistics"]["total_images"] = total_images
metadata["statistics"]["total_size_mb"] = round(total_size / (1024 * 1024), 2)

# Save metadata as JSON
json_filename = os.path.join(script_dir, "dataset_metadata.json")
try:
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Metadata saved: {json_filename}")
except Exception as e:
    print(f"Error saving JSON: {e}")

# Create a human-readable README
readme_filename = os.path.join(script_dir, "DATASET_INFO.txt")
try:
    with open(readme_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("DATASET METADATA\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated: {metadata['generated_at']}\n\n")
        
        f.write("STATISTICS:\n")
        f.write(f"  Total Images: {metadata['statistics']['total_images']}\n")
        f.write(f"  Total Size: {metadata['statistics']['total_size_mb']} MB\n")
        f.write(f"  Image Formats: {dict(metadata['statistics']['image_formats'])}\n\n")
        
        f.write("CLASSES:\n")
        for class_name, count in metadata['classes'].items():
            f.write(f"  {class_name}: {count} images\n")
        f.write("\n")
        
        f.write("SPLITS:\n")
        for split_name, split_data in metadata['splits'].items():
            f.write(f"\n  {split_name.upper()} ({split_data['total_images']} images):\n")
            for class_name, class_data in split_data['categories'].items():
                f.write(f"    {class_name}: {class_data['count']} images ({class_data['size_mb']} MB)\n")
    
    print(f"✓ README saved: {readme_filename}")
except Exception as e:
    print(f"Error saving README: {e}")

print(f"\n✓ Dataset metadata generation complete!")
print(f"Total images: {total_images}")
print(f"Total size: {metadata['statistics']['total_size_mb']} MB")
