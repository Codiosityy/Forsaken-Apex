for folder in class_folders:
    count = len(os.listdir(folder))
    if count < 7:
        print(f"Warning: {folder} has only {count} images")
```

Classes with < 7 images may not have samples in all splits due to rounding in [Preprocessing_Scripts/Seggregate_Dataset.py:45-46](../Preprocessing_Scripts/Seggregate_Dataset.py#L45-L46)

---

### Issue: Grayscale Conversion Skips Files

**Symptom**: Console output shows `"Skipped (could not read): /path/to/file"`

**Causes**:
1. Corrupted image file → OpenCV `imread()` returns `None` [Preprocessing_Scripts/grayscale_conversion.py:8](../Preprocessing_Scripts/grayscale_conversion.py#L8)
2. Unsupported format → Not in `supported_formats` set [line 14]()
3. File permissions → Cannot read file

**Diagnosis**:
```bash
# Test individual file
python -c "import cv2; print(cv2.imread('problematic_file.jpg') is None)"
```

**Solution**: Remove or repair corrupted files before conversion

---

### Issue: Metadata Generation Shows Zero Images

**Symptom**: `dataset_metadata.json` reports `"total_images": 0`

**Cause**: Script executed before `Seggregate_Dataset.py`, so `train/`, `val/`, `test/` don't exist

**Solution**: Execute preprocessing pipeline in correct order:
```bash
python Seggregate_Dataset.py      # First
python grayscale_conversion.py train/
python Dataset_Metadata_Generation.py  # Last
```

---

### Issue: Training Script Cannot Find Images

**Symptom**: `tf.keras.utils.image_dataset_from_directory()` returns empty dataset

**Causes**:
1. Incorrect path to split directory
2. Grayscale files not created (training script looks for `*_grayscale.jpg`)
3. Empty class directories

**Diagnosis**:
```python
import os
train_dir = 'Preprocessing_Scripts/train'
classes = os.listdir(train_dir)
for cls in classes:
    images = [f for f in os.listdir(os.path.join(train_dir, cls)) if f.endswith('.jpg')]
    print(f"{cls}: {len(images)} images")
```

**Solution**: Verify grayscale conversion completed and training script uses correct paths

---

### Issue: Class Imbalance in Splits

**Symptom**: `DATASET_INFO.txt` shows some classes with far fewer samples than others

**Cause**: Original dataset has severe class imbalance (e.g., only 5 'good' wafer samples)

**Solutions**:
1. **Preprocessing Level**: Not addressed by these scripts
2. **Training Level**: Use class weighting (see [Data Pipeline](#2.3))
3. **Two-Phase Training**: See [train2.py approach](#5.1) with oversampling

**Note**: Random splitting preserves original class distribution, as designed

Sources: [Preprocessing_Scripts/Seggregate_Dataset.py:44](../Preprocessing_Scripts/Seggregate_Dataset.py#L44) (random shuffle), error handling patterns across all scripts