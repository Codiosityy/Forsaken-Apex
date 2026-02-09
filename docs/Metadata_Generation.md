## Purpose and Scope

This document describes the `Dataset_Metadata_Generation.py` script, which generates comprehensive documentation of dataset composition, statistics, and class distribution after preprocessing. The script produces both machine-readable JSON and human-readable text outputs that capture dataset structure, image counts, file sizes, and format information across train/validation/test splits.

For information about the dataset organization that precedes this step, see [Dataset Organization](./Dataset_Organization_Seggregate_Dataset.py.md#3.1). For details on grayscale preprocessing, see [Grayscale Conversion](./Grayscale_Conversion.md#3.2).

**Sources:** [Preprocessing_Scripts/Dataset_Metadata_Generation.py:1-139](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L1-L139)

---

## Functional Overview

The `Dataset_Metadata_Generation.py` script serves as a post-processing documentation tool that analyzes the canonical train/val/test directory structure and generates statistical summaries. Unlike other preprocessing scripts that transform data, this script is **read-only** and non-destructive—it only inspects the dataset and creates documentation artifacts.

The script generates two complementary outputs:
1. **`dataset_metadata.json`** - Structured JSON containing detailed statistics organized by split and class
2. **`DATASET_INFO.txt`** - Human-readable text summary for quick reference

### Diagram: Metadata Generation System Architecture

```mermaid
graph TB
    subgraph Input["Input: Canonical Dataset Structure"]
        TRAIN["train/<br/>├─ class1/<br/>├─ class2/<br/>└─ classN/"]
        VAL["val/<br/>├─ class1/<br/>├─ class2/<br/>└─ classN/"]
        TEST["test/<br/>├─ class1/<br/>├─ class2/<br/>└─ classN/"]
    end
    
    subgraph Processing["Dataset_Metadata_Generation.py"]
        INIT["Initialize metadata dict<br/>{splits, classes, statistics}"]
        SCAN["Scan each split directory"]
        COUNT["Count images per class<br/>Calculate file sizes<br/>Track image formats"]
        AGGREGATE["Aggregate totals<br/>Cross-split class statistics"]
    end
    
    subgraph Output["Output Artifacts"]
        JSON["dataset_metadata.json<br/>Structured JSON"]
        TXT["DATASET_INFO.txt<br/>Human-readable summary"]
    end
    
    TRAIN --> SCAN
    VAL --> SCAN
    TEST --> SCAN
    
    SCAN --> COUNT
    COUNT --> AGGREGATE
    AGGREGATE --> JSON
    AGGREGATE --> TXT
    
    INIT --> SCAN
```

**Sources:** [Preprocessing_Scripts/Dataset_Metadata_Generation.py:6-139](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L6-L139)

---

## Input Requirements

The script expects a specific directory structure matching the output of `Seggregate_Dataset.py`:

| Directory Level | Description | Example |
|----------------|-------------|---------|
| Root | Script directory | `Preprocessing_Scripts/` |
| Split Level | `train/`, `val/`, `test/` directories | Fixed names |
| Class Level | Subdirectories named by class | `Center/`, `Edge-Loc/`, `good/`, etc. |
| Image Files | Image files with supported extensions | `*.jpg`, `*.png`, `*.gif`, `*.bmp`, `*.tiff`, `*.webp` |

The supported image extensions are defined at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:10](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L10):

```python
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
```

The script assumes it is located in the same directory as the train/val/test folders and uses its own location as the dataset root at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:7](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L7).

**Sources:** [Preprocessing_Scripts/Dataset_Metadata_Generation.py:7-26](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L7-L26)

---

## Processing Workflow

### Diagram: Metadata Collection Process Flow

```mermaid
flowchart TD
    START["Script Execution<br/>line 30"]
    INIT["Initialize metadata structure<br/>lines 13-23"]
    
    LOOP_SPLITS["For each split in ['train', 'val', 'test']<br/>lines 33-91"]
    CHECK_DIR{"Split directory exists?<br/>lines 36-38"}
    SKIP["Print warning, skip split<br/>line 37"]
    
    INIT_SPLIT["Initialize split metadata<br/>lines 40-44"]
    GET_CLASSES["List class subdirectories<br/>line 47"]
    
    LOOP_CLASSES["For each class directory<br/>lines 49-89"]
    LIST_IMAGES["Find all image files<br/>lines 53-55"]
    CHECK_EMPTY{"Images found?<br/>lines 57-58"}
    SKIP_CLASS["Continue to next class"]
    
    CALC_STATS["Calculate statistics:<br/>• File sizes<br/>• Format counts<br/>lines 60-76"]
    UPDATE_META["Update metadata:<br/>• Split totals<br/>• Class totals<br/>• Format totals<br/>lines 78-89"]
    
    SAVE_JSON["Save dataset_metadata.json<br/>lines 98-104"]
    SAVE_TXT["Save DATASET_INFO.txt<br/>lines 107-134"]
    END["Print completion message<br/>lines 136-138"]
    
    START --> INIT
    INIT --> LOOP_SPLITS
    LOOP_SPLITS --> CHECK_DIR
    CHECK_DIR -->|No| SKIP
    CHECK_DIR -->|Yes| INIT_SPLIT
    SKIP --> LOOP_SPLITS
    
    INIT_SPLIT --> GET_CLASSES
    GET_CLASSES --> LOOP_CLASSES
    LOOP_CLASSES --> LIST_IMAGES
    LIST_IMAGES --> CHECK_EMPTY
    CHECK_EMPTY -->|No| SKIP_CLASS
    CHECK_EMPTY -->|Yes| CALC_STATS
    SKIP_CLASS --> LOOP_CLASSES
    
    CALC_STATS --> UPDATE_META
    UPDATE_META --> LOOP_CLASSES
    
    LOOP_SPLITS --> SAVE_JSON
    SAVE_JSON --> SAVE_TXT
    SAVE_TXT --> END
```

**Sources:** [Preprocessing_Scripts/Dataset_Metadata_Generation.py:30-139](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L30-L139)

---

## Metadata Structure

The script maintains a nested dictionary structure that captures dataset information at multiple levels of granularity.

### Root Metadata Schema

The root metadata object is initialized at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:13-23](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L13-L23):

| Field | Type | Description |
|-------|------|-------------|
| `generated_at` | ISO timestamp | Metadata generation time |
| `dataset_root` | String | Absolute path to dataset root |
| `splits` | Object | Per-split statistics (train/val/test) |
| `classes` | Object | Cross-split class totals |
| `statistics` | Object | Global statistics |

### Split-Level Structure

Each entry in the `splits` object follows the structure defined at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:40-44](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L40-L44):

```json
{
  "path": "train",
  "total_images": 1500,
  "categories": {
    "class_name": {
      "count": 250,
      "size_mb": 12.5,
      "formats": {".jpg": 250}
    }
  }
}
```

### Diagram: Metadata JSON Hierarchy

```mermaid
graph TD
    ROOT["metadata (root object)"]
    
    ROOT --> GEN["generated_at<br/>ISO timestamp"]
    ROOT --> DROOT["dataset_root<br/>script directory path"]
    ROOT --> SPLITS["splits object"]
    ROOT --> CLASSES["classes object"]
    ROOT --> STATS["statistics object"]
    
    SPLITS --> TRAIN["train object"]
    SPLITS --> VAL["val object"]
    SPLITS --> TEST["test object"]
    
    TRAIN --> TRAIN_PATH["path: 'train'"]
    TRAIN --> TRAIN_TOTAL["total_images: N"]
    TRAIN --> TRAIN_CATS["categories object"]
    
    TRAIN_CATS --> CLASS1["class_name object"]
    CLASS1 --> C1_COUNT["count: N images"]
    CLASS1 --> C1_SIZE["size_mb: X.XX"]
    CLASS1 --> C1_FMTS["formats object<br/>{'.jpg': N, '.png': M}"]
    
    CLASSES --> CLS_TOTALS["'class_name': total_count"]
    
    STATS --> ST_TOTAL["total_images: N"]
    STATS --> ST_SIZE["total_size_mb: X.XX"]
    STATS --> ST_FMTS["image_formats object<br/>{'.jpg': N, '.png': M}"]
```

**Sources:** [Preprocessing_Scripts/Dataset_Metadata_Generation.py:13-23](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L13-L23), [Preprocessing_Scripts/Dataset_Metadata_Generation.py:40-44](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L40-L44), [Preprocessing_Scripts/Dataset_Metadata_Generation.py:72-76](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L72-L76)

---

## Image Statistics Calculation

For each class within each split, the script calculates three key metrics:

### File Count
The number of images is determined by filtering directory contents for files with supported extensions at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:53-55](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L53-L55):

```python
image_files = [f for f in os.listdir(class_path)
               if os.path.isfile(os.path.join(class_path, f))
               and os.path.splitext(f)[1].lower() in image_extensions]
```

### Size Calculation
Total storage size is computed by summing individual file sizes at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:64-66](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L64-L66):

```python
for img_file in image_files:
    img_path = os.path.join(class_path, img_file)
    class_size += os.path.getsize(img_path)
```

The size is converted to megabytes and rounded to 2 decimal places at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:74](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L74).

### Format Tracking
Image formats are tallied using a dictionary accumulator at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:68-69](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L68-L69):

```python
ext = os.path.splitext(img_file)[1].lower()
format_count[ext] = format_count.get(ext, 0) + 1
```

**Sources:** [Preprocessing_Scripts/Dataset_Metadata_Generation.py:53-76](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L53-L76)

---

## Aggregation Logic

The script maintains running totals across multiple dimensions:

### Per-Split Aggregation
Each split accumulates its own total image count at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:78](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L78):

```python
metadata["splits"][split_name]["total_images"] += len(image_files)
```

### Cross-Split Class Totals
The `classes` object tracks how many images of each class exist across all splits at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:87-89](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L87-L89):

```python
if class_name not in metadata["classes"]:
    metadata["classes"][class_name] = 0
metadata["classes"][class_name] += len(image_files)
```

This allows analysis of class distribution independent of split boundaries.

### Global Statistics
Dataset-wide totals are accumulated at [Preprocessing_Scripts/Dataset_Metadata_Generation.py:79-80](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L79-L80) and [Preprocessing_Scripts/Dataset_Metadata_Generation.py:82-84](../Preprocessing_Scripts/Dataset_Metadata_Generation.py#L82-L84):

```python
total_images += len(image_files)
total_size += class_size