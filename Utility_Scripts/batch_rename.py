import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get all files in the root folder of the script
files = os.listdir(script_dir)

# Filter for .jpg files only (case-insensitive) and ensure they are files, not directories
jpg_files = [f for f in files if f.lower().endswith('.jpg') and os.path.isfile(os.path.join(script_dir, f))]

# Sort the files for consistent ordering
jpg_files.sort()

# Rename files sequentially
for index, filename in enumerate(jpg_files, start=1):
    old_path = os.path.join(script_dir, filename)
    new_filename = f"c_{index}.jpg"
    new_path = os.path.join(script_dir, new_filename)
    
    try:
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
    except Exception as e:
        print(f"Error renaming {filename}: {e}")

print(f"Total files renamed: {len(jpg_files)}")
