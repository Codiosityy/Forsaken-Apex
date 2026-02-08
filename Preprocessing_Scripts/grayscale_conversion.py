import cv2
import os
from pathlib import Path

def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def process_folder(folder_path):
    """Process all images in a folder and its subfolders"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    processed_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in supported_formats:
                input_path = os.path.join(root, file)
                gray_image = convert_to_grayscale(input_path)
                
                if gray_image is not None:
                    # Save with "_grayscale" prefix
                    filename = Path(file).stem
                    extension = Path(file).suffix
                    output_filename = f"{filename}_grayscale{extension}"
                    output_path = os.path.join(root, output_filename)
                    
                    cv2.imwrite(output_path, gray_image)
                    print(f"Converted: {input_path}")
                    processed_count += 1
                else:
                    print(f"Skipped (could not read): {input_path}")
                    skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {processed_count}")
    print(f"Total images skipped: {skipped_count}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python grayscale.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory")
        sys.exit(1)
    
    process_folder(folder_path)   
