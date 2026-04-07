<div align="center">

# 🔬 Forsaken-Apex

**Production-ready deep learning system for automated semiconductor wafer defect detection and classification.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Wiki](https://img.shields.io/badge/Docs-Wiki-blueviolet)](https://github.com/Codiosityy/Forsaken-Apex/wiki)

</div>

---

## What is Forsaken-Apex?

Forsaken-Apex is a deep learning pipeline that automatically identifies and classifies manufacturing defects in semiconductor wafer images — defects like scratches, coating issues, contamination, etching failures, bridging, voids, and foreign material. It is designed to integrate into real-world quality control workflows, not just academic experiments.

The system uses a **MobileNetV2 backbone with Squeeze-and-Excitation (SE) attention**, trained via a **progressive resizing curriculum** (128→160→224px) and a custom **Focal Loss** to handle the severe class imbalance inherent in defect datasets. After training, a **5-test validation suite** rigorously evaluates the model beyond simple accuracy.

---

## Key Features

- **Progressive Curriculum Training** — trains at 128×128, 160×160, then 224×224 resolution, progressively unlocking backbone layers for stable fine-tuning
- **Focal Loss** — down-weights easy negatives so the model focuses on hard, rare defect classes
- **SE Attention** — channel-wise feature recalibration via Squeeze-and-Excitation blocks, improving sensitivity to subtle defect patterns
- **Dual-mode execution** — identical implementation in both `kaggle-notebook.ipynb` (interactive) and `train.py` (production script)
- **5-test evaluation suite** — goes beyond accuracy with calibration analysis, perturbation robustness, entropy-based uncertainty, and generalization gap testing

---

## Quick Start

### 1. Prepare your dataset

```bash
# Organize raw images into train/val/test splits (70/15/15)
python Preprocessing_Scripts/Seggregate_Dataset.py

# Convert images to grayscale
python Preprocessing_Scripts/grayscale_conversion.py
```

### 2. Train the model

```bash
# Production script
python train.py

# Or use the Kaggle notebook interactively
jupyter notebook kaggle-notebook.ipynb
```

### 3. Evaluate

```bash
python Evaluate_model.py --model path/to/final_model.keras
```

This runs all 5 validation tests and produces a `validation_report.json` with an overall verdict (`VALIDATED` / `NOT VALIDATED`) and 5 diagnostic PNG plots.

---

## Repository Structure

```
Forsaken-Apex/
├── train.py                        # Production training script
├── kaggle-notebook.ipynb           # Interactive development notebook
├── Evaluate_model.py               # 5-test model evaluation suite
│
├── Preprocessing_Scripts/
│   ├── Seggregate_Dataset.py       # Train/val/test split (70/15/15)
│   ├── grayscale_conversion.py     # RGB → grayscale conversion
│   └── Dataset_Metadata_Generation.py
│
├── Utility_Scripts/
│   ├── load_dataset.py             # LSWMD.pkl dataset inspection
│   ├── extract_pdf_images.py       # Research figure extraction
│   ├── batch_rename.py             # Bulk file renaming
│   └── random_file_picker.py       # Random dataset sampling
│
├── Previous_training_scripts/      # Deprecated approaches (kept for reference)
│   ├── train1.py                   # CNN-SVM ensemble (deprecated)
│   └── train2.py                   # Two-phase class-balanced (deprecated)
│
├── Evaluation_Results/             # Output from evaluation runs
└── docs/                           # Additional documentation assets
```

---

## How It Works

### Training Architecture

| Component | Detail |
|---|---|
| Backbone | MobileNetV2 (`alpha=0.75`), ImageNet pre-trained |
| Attention | SEBlock (Squeeze-and-Excitation, ratio=16) |
| Loss | FocalLoss (`γ=1.5`, `α=0.25`, label smoothing=0.1) |
| Augmentation | Random flip, rot90, MixUp (`α=0.2`) |
| Normalization | Rescaling to `[-1, 1]` via Rescaling layer |
| Progressive sizes | 128px (20 epochs) → 160px (25 epochs) → 224px (35 epochs) |
| Final fine-tuning | Top 20 backbone layers unfrozen, LR reduced 10× |

### Evaluation Tests

| Test | What it checks |
|---|---|
| Test 1: Feature Learning | Confusion matrix — per-class accuracy and semantic confusion patterns |
| Test 2: Generalization Gap | Train vs. test consistency — detects overfitting |
| Test 3: Confidence Calibration | Are high-confidence predictions actually correct? |
| Test 4: Perturbation Robustness | Accuracy retention under Gaussian noise at 3 levels |
| Test 5: Entropy Analysis | Uncertainty ratio — are wrong predictions also uncertain? |

---

## Output Artifacts

After training, the following are saved to `{OUTPUT_DIR}/`:

- `models/stage_128.keras`, `stage_160.keras`, `stage_224.keras` — best checkpoint per stage
- `models/final_model.keras` — final production model
- `results/metrics.json` — accuracy, precision, recall

After evaluation:

- `validation_report.json` — overall score and verdict
- `test1_confusion_matrix.png` through `test5_entropy_analysis.png`

---

## Documentation

The full technical documentation lives in the **[Wiki](https://github.com/Codiosityy/Forsaken-Apex/wiki)**. Start here:

| Section | Description |
|---|---|
| [Overview](https://github.com/Codiosityy/Forsaken-Apex/wiki/Overview) | System architecture, data flow, design decisions |
| [Core Training System](https://github.com/Codiosityy/Forsaken-Apex/wiki/Core_Training_System) | `ProgressiveTrainer`, `DataPipeline`, `FocalLoss`, `SEBlock` internals |
| [Model Architecture](https://github.com/Codiosityy/Forsaken-Apex/wiki/Model_Architecture) | MobileNetV2 + SE attention design |
| [Data Preprocessing Pipeline](https://github.com/Codiosityy/Forsaken-Apex/wiki/Data_Preprocessing_Pipeline) | Dataset splitting, grayscale conversion, metadata |
| [Model Evaluation Framework](https://github.com/Codiosityy/Forsaken-Apex/wiki/Model_Evaluation_Framework) | 5-test suite in detail |
| [Alternative Approaches](https://github.com/Codiosityy/Forsaken-Apex/wiki/Alternative_Training_Approaches) | Deprecated CNN-SVM and two-phase methods |
| [Utility Tools](https://github.com/Codiosityy/Forsaken-Apex/wiki/Utility_Tools) | Helper scripts for dataset inspection and management |

> New here? Start with the [Overview](https://github.com/Codiosityy/Forsaken-Apex/wiki/Overview) for a full walkthrough of how the system fits together.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.
