## Purpose and Scope

This document describes the two-phase training approach implemented in `train2.py`, an alternative training strategy that addresses extreme class imbalance in the wafer defect dataset through Class-Balanced Loss and staged training. This approach is particularly designed to handle cases where minority classes (such as the 'good' class with only 5 samples) are severely underrepresented.

This system is distinct from the current production approach (see [2](#2)) which uses progressive resizing and FocalLoss. For the CNN-SVM ensemble alternative, see [5.2](#5.2).

**Sources:** [Previous_training_scripts/train2.py:1-534](../Previous_training_scripts/train2.py#L1-L534)

---

## System Architecture Overview

The `train2.py` system employs a fundamentally different strategy than the current production system, focusing on data-level solutions (oversampling) combined with loss-function-based reweighting, followed by adaptation to the real-world distribution through fine-tuning.

```mermaid
flowchart TB
    subgraph Input["Data Input"]
        TRAIN["TRAIN_DIR<br/>/Segregated_defects_grayscale/train"]
        TEST["TEST_DIR<br/>/Segregated_defects_grayscale/test"]
    end
    
    subgraph Analysis["Step 1: Analysis"]
        ANALYZE["analyze_class_distribution()<br/>Count samples per class<br/>Identify imbalance ratio"]
    end
    
    subgraph Resampling["Step 2: Data Preparation"]
        BALANCE["create_balanced_dataset()<br/>Oversample to 500/class<br/>Repeat minority classes"]
        AUG_MIN["Minority Class Aug<br/>AGGRESSIVE<br/>rotation=0.5, zoom=0.2"]
        AUG_MAJ["Majority Class Aug<br/>CONSERVATIVE<br/>rotation=0.25, zoom=0.0"]
    end
    
    subgraph Loss["Step 3: Loss Function"]
        CBLOSS["ClassBalancedLoss<br/>beta=0.9999<br/>loss_type='focal'<br/>gamma=2.0"]
        WEIGHTS["Compute weights<br/>(1-β^n)/(1-β)"]
    end
    
    subgraph Phase1["Step 4: Phase 1 Training"]
        P1_DATA["train_ds_balanced<br/>Oversampled data"]
        P1_OPT["AdamW<br/>LR=1e-3<br/>weight_decay=1e-4"]
        P1_TRAIN["EPOCHS_PHASE1=30<br/>Monitor val_f1<br/>EarlyStopping patience=10"]
        P1_SAVE["best_balanced.keras"]
    end
    
    subgraph Phase2["Step 5: Phase 2 Fine-tuning"]
        P2_DATA["train_ds_full<br/>Original imbalanced data"]
        P2_OPT["AdamW<br/>LR=1e-5<br/>weight_decay=1e-4"]
        P2_TRAIN["EPOCHS_PHASE2=20<br/>Monitor val_f1<br/>EarlyStopping patience=15"]
        P2_SAVE["best_finetuned.keras"]
    end
    
    subgraph Threshold["Step 6: Threshold Optimization"]
        FIND_THRESH["find_optimal_threshold()<br/>Per-class F1 maximization<br/>Search range: 0.1-0.9"]
    end
    
    subgraph Eval["Step 7: Evaluation"]
        METRICS["evaluate_with_proper_metrics()<br/>Classification report<br/>Confusion matrix<br/>Per-class P/R/F1"]
    end
    
    TRAIN --> ANALYZE
    TEST --> ANALYZE
    ANALYZE --> BALANCE
    BALANCE --> AUG_MIN
    BALANCE --> AUG_MAJ
    
    ANALYZE --> CBLOSS
    CBLOSS --> WEIGHTS
    
    BALANCE --> P1_DATA
    WEIGHTS --> P1_DATA
    P1_DATA --> P1_OPT
    P1_OPT --> P1_TRAIN
    P1_TRAIN --> P1_SAVE
    
    P1_SAVE --> P2_DATA
    WEIGHTS --> P2_DATA
    P2_DATA --> P2_OPT
    P2_OPT --> P2_TRAIN
    P2_TRAIN --> P2_SAVE
    
    P2_SAVE --> FIND_THRESH
    FIND_THRESH --> METRICS
    METRICS --> FINAL["wafer_classifier_imbalanced.keras"]
```

**Sources:** [Previous_training_scripts/train2.py:1-534](../Previous_training_scripts/train2.py#L1-L534)

---

## Configuration and Hyperparameters

The system uses fixed configuration parameters that differ significantly from the production system:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `IMG_SIZE` | 224 | Fixed resolution (no progressive resizing) |
| `BATCH_SIZE` | 32 | Standard batch size |
| `EPOCHS_PHASE1` | 30 | Balanced dataset training duration |
| `EPOCHS_PHASE2` | 20 | Fine-tuning duration on full data |
| `target_samples_per_class` | 500 | Oversampling target |
| `beta` | 0.9999 | Class-Balanced Loss parameter |
| `gamma` | 2.0 | Focal loss focusing parameter |

The configuration is defined at [Previous_training_scripts/train2.py:10-13](../Previous_training_scripts/train2.py#L10-L13) and referenced throughout the training pipeline. Unlike the production system, there is no progressive resizing curriculum—all training occurs at 224×224 resolution.

**Sources:** [Previous_training_scripts/train2.py:10-16](../Previous_training_scripts/train2.py#L10-L16)

---

## Class Distribution Analysis

The `analyze_class_distribution()` function provides critical diagnostic information about the dataset imbalance before training begins:

```mermaid
flowchart LR
    DIR["directory<br/>(TRAIN_DIR or TEST_DIR)"]
    
    subgraph Scan["Scanning Phase"]
        LIST["os.listdir(directory)<br/>Enumerate class folders"]
        COUNT["Count images per class<br/>Filter: .png/.jpg/.jpeg/.bmp/.tiff"]
    end
    
    subgraph Analysis["Analysis Phase"]
        TOTAL["Calculate total samples"]
        PERCENT["Compute percentage per class"]
        RATIO["Compute imbalance ratio<br/>max_count/min_count"]
    end
    
    subgraph Output["Console Output"]
        BAR["Visual bar chart<br/>█ symbols scaled to 2%"]
        STATS["Min/Max counts<br/>Imbalance ratio"]
    end
    
    DIR --> LIST
    LIST --> COUNT
    COUNT --> TOTAL
    TOTAL --> PERCENT
    COUNT --> RATIO
    
    PERCENT --> BAR
    RATIO --> STATS
```

The function implements a visualization using ASCII bar charts where each █ character represents approximately 2% of the dataset. This provides immediate visual feedback about the severity of class imbalance. The imbalance ratio (e.g., "Most imbalanced ratio: 1000:1") quantifies the disparity between the most and least represented classes.

**Example output structure:**
```
📊 CLASS DISTRIBUTION ANALYSIS:
============================================================
good                     :     5 ( 0.05%) █
Edge-Loc                 :   347 ( 3.47%) ██████
Center                   :   450 ( 4.50%) █████████
...
============================================================
Total images: 10000
Most imbalanced ratio: 2000.0:1
```

**Sources:** [Previous_training_scripts/train2.py:21-46](../Previous_training_scripts/train2.py#L21-L46)

---

## Class-Balanced Loss Function

The `ClassBalancedLoss` class implements the loss function from "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019), which addresses imbalance by reweighting based on the effective number of samples rather than raw counts.

### Mathematical Formulation

The loss computes class weights using the effective number formula:

```
effective_num = (1 - β^n) / (1 - β)
weight = 1 / effective_num
```

Where:
- `n` is the number of samples in the class
- `β` is a hyperparameter (0.9999) that controls reweighting strength
- Higher `β` values approach inverse frequency weighting

```mermaid
graph TD
    subgraph Input["Constructor Inputs"]
        SAMPLES["samples_per_class<br/>Array of sample counts"]
        BETA["beta=0.9999<br/>Reweighting strength"]
        LOSS_TYPE["loss_type='focal'<br/>or 'crossentropy'"]
        GAMMA["gamma=2.0<br/>Focal loss parameter"]
    end
    
    subgraph Computation["Weight Computation"]
        EFFECTIVE["effective_num = (1-β^n)/(1-β)<br/>for each class"]
        INVERT["weights = 1 / effective_num"]
        NORMALIZE["Normalize: sum(weights) = n_classes"]
    end
    
    subgraph Call["call(y_true, y_pred)"]
        EXTRACT["Extract per-sample weights<br/>weights * y_true"]
        BRANCH{loss_type?}
        CE["Categorical Cross-Entropy<br/>weighted by class weights"]
        FOCAL["Focal Loss Component<br/>p_t = sum(y_true * y_pred)<br/>focal_weight = (1 - p_t)^gamma"]
        COMBINE["weighted * focal_weight * ce"]
    end
    
    SAMPLES --> EFFECTIVE
    BETA --> EFFECTIVE
    EFFECTIVE --> INVERT
    INVERT --> NORMALIZE
    
    NORMALIZE --> EXTRACT
    EXTRACT --> BRANCH
    BRANCH -->|"crossentropy"| CE
    BRANCH -->|"focal"| FOCAL
    FOCAL --> COMBINE
```

### Implementation Details

The `ClassBalancedLoss.__init__()` method [Previous_training_scripts/train2.py:59-76](../Previous_training_scripts/train2.py#L59-L76) precomputes class weights during initialization, printing diagnostic output showing the sample count and computed weight for each class. This allows verification that minority classes receive appropriately higher weights.

The `call()` method [Previous_training_scripts/train2.py:78-95](../Previous_training_scripts/train2.py#L78-L95) applies these weights to either standard cross-entropy or focal loss. When `loss_type="focal"`, it combines:
1. Class-based reweighting from effective number of samples
2. Prediction-based reweighting from focal loss focusing parameter

This dual reweighting addresses both dataset-level imbalance (via class weights) and prediction-level difficulty (via focal loss).

**Sources:** [Previous_training_scripts/train2.py:51-96](../Previous_training_scripts/train2.py#L51-L96)

---

## Balanced Dataset Creation via Oversampling

The `create_balanced_dataset()` function implements aggressive oversampling to equalize class representation before training.

```mermaid
flowchart TD
    subgraph Input["Inputs"]
        DIR["directory<br/>TRAIN_DIR"]
        TARGET["target_samples_per_class=500"]
    end
    
    subgraph Scan["Class Scanning"]
        ENUMERATE["Enumerate class directories"]
        COLLECT["Collect image paths per class"]
    end
    
    subgraph Calculate["Repeat Factor Calculation"]
        COMPUTE["repeat_factor = max(1, 500 // n_samples)<br/>For each class"]
        EXAMPLE1["Example: 'good' with 5 samples<br/>repeat_factor = 500 // 5 = 100"]
        EXAMPLE2["Example: 'Edge-Ring' with 400 samples<br/>repeat_factor = 500 // 400 = 1"]
    end
    
    subgraph Generate["Dataset Generation"]
        REPEAT["Repeat each image path<br/>repeat_factor times"]
        CREATE["Create all_files[] and all_labels[]"]
        MAP["Map class_name to numeric index"]
    end
    
    subgraph Output["Outputs"]
        FILES["all_files: List of image paths"]
        LABELS["all_labels: List of class indices"]
        NAMES["class_names: Sorted class list"]
    end
    
    DIR --> ENUMERATE
    TARGET --> COMPUTE
    ENUMERATE --> COLLECT
    COLLECT --> COMPUTE
    COMPUTE --> EXAMPLE1
    COMPUTE --> EXAMPLE2
    
    COMPUTE --> REPEAT
    REPEAT --> CREATE
    CREATE --> MAP
    
    MAP --> FILES
    MAP --> LABELS
    MAP --> NAMES
```

### Oversampling Strategy

The function [Previous_training_scripts/train2.py:100-141](../Previous_training_scripts/train2.py#L100-L141) implements simple repetition of file paths rather than immediate data duplication. For a minority class like 'good' with 5 samples:
- `repeat_factor = 500 // 5 = 100`
- Each of the 5 images appears 100 times in `all_files`
- Total: ~500 samples for the 'good' class

This approach defers augmentation to training time, where each repetition will receive different random augmentations, creating synthetic diversity. The repeated paths consume minimal memory compared to storing actual augmented images.

**Sources:** [Previous_training_scripts/train2.py:100-141](../Previous_training_scripts/train2.py#L100-L141)

---

## Differential Augmentation Strategy

The `create_augmentation_pipeline()` function creates different augmentation strategies based on class prevalence:

| Augmentation Type | Minority Classes | Majority Classes |
|-------------------|------------------|------------------|
| Flip | horizontal_and_vertical | horizontal only |
| Rotation | 0.5 (±180°) | 0.25 (±90°) |
| Zoom | 0.2 (±20%) | None |
| Translation | 0.2 (±20%) | None |
| Brightness | 0.3 (±30%) | 0.1 (±10%) |
| Contrast | 0.3 (±30%) | 0.1 (±10%) |

**Rationale:** Minority classes need more aggressive augmentation to create synthetic diversity since each base image is repeated many times (e.g., 100× for the 'good' class). Majority classes need only conservative augmentation to prevent overfitting while maintaining defect pattern integrity.

**Sources:** [Previous_training_scripts/train2.py:146-169](../Previous_training_scripts/train2.py#L146-L169)

---

## Two-Phase Training Strategy

The core training philosophy divides learning into two distinct phases with different objectives:

```mermaid
stateDiagram-v2
    [*] --> Phase1Setup
    
    state "Phase 1: Balanced Training" as Phase1 {
        Phase1Setup: Dataset: train_ds_balanced (oversampled)
        Phase1Setup: Loss: ClassBalancedLoss(β=0.9999, focal)
        Phase1Setup: Optimizer: AdamW(LR=1e-3, WD=1e-4)
        Phase1Setup: Epochs: 30
        Phase1Setup: Objective: Learn all classes equally
        
        Phase1Setup --> Phase1Train
        Phase1Train --> Phase1Monitor
        Phase1Monitor: Monitor: val_f1
        Phase1Monitor: EarlyStopping: patience=10
        Phase1Monitor: ReduceLROnPlateau: factor=0.5, patience=5
        Phase1Monitor --> Phase1Save
        Phase1Save: Save: best_balanced.keras
    }
    
    Phase1Save --> Phase2Setup
    
    state "Phase 2: Fine-tuning" as Phase2 {
        Phase2Setup: Dataset: train_ds_full (original imbalanced)
        Phase2Setup: Loss: ClassBalancedLoss (same)
        Phase2Setup: Optimizer: AdamW(LR=1e-5, WD=1e-4)
        Phase2Setup: Epochs: 20
        Phase2Setup: Objective: Adapt to real distribution
        
        Phase2Setup --> Phase2Train
        Phase2Train --> Phase2Monitor
        Phase2Monitor: Monitor: val_f1
        Phase2Monitor: EarlyStopping: patience=15
        Phase2Monitor: ReduceLROnPlateau: factor=0.3, patience=7
        Phase2Monitor --> Phase2Save
        Phase2Save: Save: best_finetuned.keras
    }
    
    Phase2Save --> [*]
```

### Phase 1: Balanced Training

**Objective:** Teach the model to recognize all classes, especially minorities.

The `two_phase_training()` function's Phase 1 implementation [Previous_training_scripts/train2.py:192-243](../Previous_training_scripts/train2.py#L192-L243) uses:
- **Dataset:** `train_ds_balanced` with oversampled data (~500 samples per class)
- **Loss:** `ClassBalancedLoss` with `beta=0.9999` and `loss_type="focal"`
- **Learning Rate:** `1e-3` (standard initial LR for feature learning)
- **Metrics:** `['accuracy', 'precision', 'recall', 'f1']` with F1 as primary monitor
- **Callbacks:**
  - `EarlyStopping(monitor='val_f1', patience=10)`: Stop if no F1 improvement
  - `ReduceLROnPlateau(factor=0.5, patience=5)`: Halve LR if F1 plateaus
  - `ModelCheckpoint('best_balanced.keras', monitor='val_f1')`: Save best model

**Key Insight:** By training on balanced data first, the model learns feature representations for minority classes that would otherwise be ignored.

### Phase 2: Fine-tuning on Full Dataset

**Objective:** Adapt the learned representations to the real-world class distribution.

Phase 2 implementation [Previous_training_scripts/train2.py:245-289](../Previous_training_scripts/train2.py#L245-L289) differs in:
- **Dataset:** `train_ds_full` with original imbalanced distribution
- **Learning Rate:** `1e-5` (100× lower to prevent catastrophic forgetting)
- **Patience:** Increased to 15 epochs to allow gradual adaptation
- **Loss:** Still uses `ClassBalancedLoss` to prevent complete bias toward majority

**Key Insight:** The lower learning rate and class-balanced loss prevent the model from "forgetting" minority class features while adjusting prediction distributions to match real-world prevalence.

**Sources:** [Previous_training_scripts/train2.py:186-291](../Previous_training_scripts/train2.py#L186-L291)

---

## Optimal Threshold Finding

Unlike the standard argmax classification, `find_optimal_threshold()` computes per-class thresholds that maximize F1 score:

```mermaid
flowchart TD
    subgraph Input["Inputs"]
        MODEL["model: Trained classifier"]
        VAL_DS["val_ds: Validation dataset"]
        CLASSES["class_names: List of class labels"]
    end
    
    subgraph Collection["Prediction Collection"]
        ITERATE["Iterate through val_ds batches"]
        PREDICT["model.predict(images)"]
        STORE_PRED["y_scores: All predictions"]
        STORE_TRUE["y_true: True one-hot labels"]
    end
    
    subgraph PerClass["Per-Class Threshold Search"]
        LOOP["For each class i in 0..n_classes-1"]
        BINARY["Convert to binary problem<br/>y_true_binary = (argmax(y_true) == i)"]
        SEARCH["For thresh in [0.1, 0.15, ..., 0.85, 0.9]"]
        THRESHOLD["y_pred_binary = (y_scores[:, i] >= thresh)"]
        COMPUTE["Compute precision, recall, F1"]
        TRACK["Track best_thresh with highest F1"]
    end
    
    subgraph Output["Output"]
        DICT["optimal_thresholds: Dict[class_name -> threshold]"]
        PRINT["Print class-wise thresholds and F1 scores"]
    end
    
    MODEL --> ITERATE
    VAL_DS --> ITERATE
    ITERATE --> PREDICT
    PREDICT --> STORE_PRED
    PREDICT --> STORE_TRUE
    
    STORE_PRED --> LOOP
    STORE_TRUE --> LOOP
    CLASSES --> LOOP
    LOOP --> BINARY
    BINARY --> SEARCH
    SEARCH --> THRESHOLD
    THRESHOLD --> COMPUTE
    COMPUTE --> TRACK
    
    TRACK --> DICT
    DICT --> PRINT
```

### Threshold Search Algorithm

The implementation [Previous_training_scripts/train2.py:296-345](../Previous_training_scripts/train2.py#L296-L345) performs an exhaustive grid search:

1. **Binary decomposition:** For each class, convert the multi-class problem to binary (class i vs. rest)
2. **Grid search:** Test thresholds from 0.1 to 0.9 in steps of 0.05
3. **F1 computation:** For each threshold, compute TP, FP, FN and derive F1 score
4. **Selection:** Choose threshold with maximum F1 for each class

**Example output:**
```
🔍 Finding optimal thresholds...
  good                : threshold=0.40, F1=0.823
  Center              : threshold=0.50, F1=0.891
  Edge-Loc            : threshold=0.65, F1=0.745
```

Lower thresholds (e.g., 0.40 for 'good') make the model more sensitive to that class, reducing false negatives at the cost of potential false positives.

**Sources:** [Previous_training_scripts/train2.py:296-345](../Previous_training_scripts/train2.py#L296-L345)

---

## Evaluation with Proper Metrics

The `evaluate_with_proper_metrics()` function emphasizes precision, recall, and F1 rather than accuracy, which is misleading for imbalanced datasets:

```mermaid
flowchart LR
    subgraph Inputs["Inputs"]
        MODEL_EVAL["model: Trained model"]
        TEST_DS_EVAL["test_ds: Test dataset"]
        THRESH["optimal_thresholds<br/>(optional)"]
    end
    
    subgraph Prediction["Prediction Phase"]
        PRED_LOOP["For each test batch"]
        SCORES["Get prediction scores"]
        APPLY{Use thresholds?}
        CUSTOM["Custom threshold logic<br/>Check score >= threshold[class]<br/>Select max among qualified"]
        ARGMAX["Standard argmax<br/>np.argmax(scores)"]
    end
    
    subgraph Metrics["Metric Computation"]
        REPORT["classification_report()<br/>Per-class P/R/F1"]
        CM["confusion_matrix()<br/>Misclassification patterns"]
        SUPPORT["precision_recall_fscore_support()<br/>Weighted averages"]
    end
    
    subgraph Special["Special Minority Class Check"]
        GOOD{Is 'good' in classes?}
        CHECK_RECALL["Check good_recall >= 0.5"]
        WARN["⚠️ WARNING: Low recall for 'good' class"]
    end
    
    subgraph Output["Output"]
        PRINT_REPORT["Print classification report"]
        PRINT_TABLE["Print per-class table"]
        PRINT_CM["Print confusion matrix"]
        RETURN["Return dict with P/R/F1/support"]
    end
    
    MODEL_EVAL --> PRED_LOOP
    TEST_DS_EVAL --> PRED_LOOP
    PRED_LOOP --> SCORES
    SCORES --> APPLY
    THRESH --> APPLY
    
    APPLY -->|Yes| CUSTOM
    APPLY -->|No| ARGMAX
    
    CUSTOM --> REPORT
    ARGMAX --> REPORT
    REPORT --> CM
    CM --> SUPPORT
    
    SUPPORT --> GOOD
    GOOD -->|Yes| CHECK_RECALL
    CHECK_RECALL -->|recall < 0.5| WARN
    
    REPORT --> PRINT_REPORT
    SUPPORT --> PRINT_TABLE
    CM --> PRINT_CM
    PRINT_TABLE --> RETURN
```

### Custom Threshold Application

When `optimal_thresholds` is provided [Previous_training_scripts/train2.py:369-384](../Previous_training_scripts/train2.py#L369-L384), the function uses custom logic:
```python
for score in scores:
    pred_class = None
    max_score = 0
    for i, class_name in enumerate(class_names):
        if score[i] >= optimal_thresholds[class_name] and score[i] > max_score:
            max_score = score[i]
            pred_class = i
    if pred_class is None:
        pred_class = np.argmax(score)  # Fallback to argmax
```

This selects the highest-scoring class among those that exceed their optimal threshold, with argmax as fallback if no class qualifies.

### Minority Class Warning System

The function specifically checks the 'good' class performance [Previous_training_scripts/train2.py:418-428](../Previous_training_scripts/train2.py#L418-L428) and issues a warning if recall < 0.5, indicating the model is biased toward predicting defects. This is critical for wafer screening where false negatives (missing good wafers) are costly.

**Sources:** [Previous_training_scripts/train2.py:350-435](../Previous_training_scripts/train2.py#L350-L435)

---

## Synthetic Data Generation

The `SyntheticDataGenerator` class implements a SMOTE-style approach for images:

```mermaid
graph TD
    subgraph Init["__init__(image_paths, labels, batch_size, target_samples)"]
        CALC["Calculate repeat_factor<br/>= target_samples // n_original"]
        EXPAND["Expand paths and labels<br/>by repeat_factor"]
        SHUFFLE["Shuffle expanded dataset<br/>np.random.permutation"]
        AUG_DEF["Define augmentation pipeline<br/>RandomFlip, Rotation(0.5)<br/>Zoom(0.3), Translation(0.3)<br/>Brightness(0.4), Contrast(0.4)"]
    end
    
    subgraph GetItem["__getitem__(idx)"]
        SLICE["Slice batch_paths and batch_labels"]
        LOAD["For each path:<br/>tf.io.read_file<br/>decode_image<br/>resize to IMG_SIZE"]
        NORMALIZE["Cast to float32 / 255.0"]
        STACK["tf.stack to create batch tensor"]
        ONEHOT["One-hot encode labels"]
        AUG_APPLY["Apply augmentation<br/>self.augment(batch, training=True)"]
    end
    
    CALC --> EXPAND
    EXPAND --> SHUFFLE
    SHUFFLE --> AUG_DEF
    
    SLICE --> LOAD
    LOAD --> NORMALIZE
    NORMALIZE --> STACK
    STACK --> ONEHOT
    ONEHOT --> AUG_APPLY
    
    AUG_APPLY --> RETURN["Return (augmented_images, labels)"]
```

### Implementation Characteristics

The generator [Previous_training_scripts/train2.py:440-498](../Previous_training_scripts/train2.py#L440-L498) inherits from `tf.keras.utils.Sequence` for proper integration with Keras training:

- **`__len__()`:** Returns `len(expanded_paths) // batch_size`, enabling proper epoch iteration
- **`__getitem__(idx)`:** Generates one batch on-demand with:
  1. Image loading from disk
  2. Resizing to `IMG_SIZE` (224×224)
  3. Normalization to [0, 1]
  4. Heavy augmentation with rotation=0.5, zoom=0.3, brightness=0.4

This approach generates synthetic diversity at training time without storing augmented images, essential when oversampling 5 images 100× to create 500 variations.

**Sources:** [Previous_training_scripts/train2.py:440-498](../Previous_training_scripts/train2.py#L440-L498)

---

## Comparison with Current Production Approach

| Aspect | train2.py (This System) | Current Production ([2](#2)) |
|--------|-------------------------|------------------------------|
| **Core Strategy** | Two-phase training with oversampling | Progressive resizing curriculum |
| **Loss Function** | `ClassBalancedLoss` with β=0.9999 | `FocalLoss` with γ=1.5 |
| **Data Handling** | Oversample minority to 500/class | Class weighting + MixUp |
| **Training Phases** | Phase 1 (balanced) → Phase 2 (full) | 128px → 160px → 224px progressive |
| **Learning Rates** | 1e-3 → 1e-5 (two phases) | Cosine decay per stage |
| **Augmentation** | Differential (aggressive for minorities) | Uniform MixUp + geometric |
| **Architecture** | Not specified (likely MobileNetV2) | MobileNetV2 α=0.75 + SEBlock |
| **Threshold** | Per-class optimal thresholds | Standard argmax |
| **Primary Metric** | F1 score (weighted) | Validation accuracy |
| **Saved Models** | `wafer_classifier_imbalanced.keras` | `final_model.keras` |

### Architectural Philosophy

**train2.py philosophy:** Address imbalance at the data level (oversampling) and loss level (class-balanced weighting), then adapt to reality through fine-tuning.

**Current production philosophy:** Address imbalance through curriculum learning (progressive resizing), advanced loss function (focal), and architectural improvements (SE attention).

The current production system evolved from this approach, suggesting that the progressive training and architectural enhancements proved more effective than data-level solutions in practice.

**Sources:** [Previous_training_scripts/train2.py:1-534](../Previous_training_scripts/train2.py#L1-L534)

---

## Integration Points and Dependencies

### File System Dependencies

```mermaid
graph LR
    subgraph Inputs["Input Directories"]
        TRAIN_IN["/kaggle/input/data1423/<br/>Segregated_defects_grayscale/train"]
        TEST_IN["/kaggle/input/data1423/<br/>Segregated_defects_grayscale/test"]
    end
    
    subgraph Script["train2.py"]
        ANALYZE_FUNC["analyze_class_distribution()"]
        BALANCE_FUNC["create_balanced_dataset()"]
        TRAIN_FUNC["two_phase_training()"]
        THRESH_FUNC["find_optimal_threshold()"]
        EVAL_FUNC["evaluate_with_proper_metrics()"]
    end
    
    subgraph Outputs["Output Artifacts"]
        BAL["best_balanced.keras<br/>Phase 1 checkpoint"]
        FINE["best_finetuned.keras<br/>Phase 2 checkpoint"]
        FINAL["wafer_classifier_imbalanced.keras<br/>Final model"]
    end
    
    TRAIN_IN --> ANALYZE_FUNC
    TEST_IN --> ANALYZE_FUNC
    ANALYZE_FUNC --> BALANCE_FUNC
    BALANCE_FUNC --> TRAIN_FUNC
    TRAIN_FUNC --> THRESH_FUNC
    THRESH_FUNC --> EVAL_FUNC
    
    TRAIN_FUNC --> BAL
    TRAIN_FUNC --> FINE
    EVAL_FUNC --> FINAL
```

### External Dependencies

```python