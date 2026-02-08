import pickle
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore', category=DeprecationWarning)

# Custom unpickler to handle old pandas objects
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Remap old pandas module names to new locations
        if 'pandas.indexes' in module:
            module = module.replace('pandas.indexes', 'pandas.core.indexes')
        return super().find_class(module, name)

try:
    with open("LSWMD.pkl", 'rb') as f:
        unpickler = CustomUnpickler(f, encoding='latin-1')
        dataset = unpickler.load()
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading pickle: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Display dataset info
print(f"Dataset type: {type(dataset)}")
if hasattr(dataset, '__len__'):
    print(f"Dataset length: {len(dataset)}")

# Display first row
print("\nFirst row:")
first_row = dataset[0] if isinstance(dataset, (list, tuple)) else dataset.iloc[0]
print(first_row)

# Get wafer map and visualize
if isinstance(first_row, dict):
    wafer_map = first_row.get('waferMap')
    failure_type = first_row.get('failureType', 'Unknown')
else:
    # Access pandas Series by column name
    wafer_map = first_row['waferMap']
    failure_type = first_row['failureType']

print(f"\nWafer map type: {type(wafer_map)}")
print(f"Wafer map shape: {np.array(wafer_map).shape}")

plt.figure(figsize=(8, 8))
plt.imshow(wafer_map, cmap='gray')
plt.colorbar(label='Pixel Value')
plt.title(f"Wafer Map - Failure Type: {failure_type}")
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.savefig('wafermap_visualization.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nVisualization saved as 'wafermap_visualization.png'")
