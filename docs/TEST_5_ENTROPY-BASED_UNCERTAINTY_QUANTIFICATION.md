Entropy Statistics:
  Mean entropy (correct predictions):     0.23 ± 0.15
  Mean entropy (incorrect predictions):   0.67 ± 0.32
  Uncertainty Ratio (incorrect/correct):  2.91

Threshold-Based Analysis (threshold = 0.5):
  High-certainty predictions (< 0.5):     1847 samples
  High-certainty accuracy:                93.4%
  Low-certainty predictions (>= 0.5):     203 samples
  
✓ Model demonstrates strong uncertainty discrimination
  (Incorrect predictions have 191% higher entropy)

Per-Class Entropy:
Class                    Mean Entropy       Std Dev      Samples
------------------------------------------------------------------------
block etch                      0.18          0.12          256
bridge                          0.21          0.14          312
...
```

**Key Metrics Interpretation:**

| Metric | Desirable Range | Significance |
|--------|----------------|--------------|
| Uncertainty Ratio | > 1.5 | Incorrect predictions show clearly elevated entropy |
| High-certainty accuracy | > 90% | Low-entropy predictions are reliable |
| Mean entropy (correct) | < 0.4 | Model confident on correct classifications |

**Sources:** [Evaluate_model.py:454-487](../Evaluate_model.py#L454-L487)

---

## Integration with Evaluation Framework

### Position in Test Suite Execution

```mermaid
graph TD
    subgraph "Main Evaluation Flow"
        LOAD["load_model()<br/>line 609"]
        TESTDS["create test_ds<br/>line 619"]
    end
    
    subgraph "Test Execution Order"
        T1["TEST 1: Confusion Matrix<br/>line 631"]
        T2["TEST 2: Generalization Gap<br/>line 640"]
        T3["TEST 3: Confidence Calibration<br/>line 647"]
        T4["TEST 4: Noise Robustness<br/>line 654"]
        T5["TEST 5: Entropy Analysis<br/>line 661"]
    end
    
    subgraph "Shared Data Dependencies"
        PROBS["y_probs<br/>collected line 624-627"]
        TRUE["y_true<br/>collected line 624-627"]
        PRED["y_pred<br/>collected line 624-627"]
    end
    
    subgraph "Final Aggregation"
        REPORT["validation_report.json<br/>line 669"]
        OVERALL["overall_score<br/>weighted average"]
        VERDICT["VALIDATED LEARNING MODEL"]
    end
    
    LOAD --> TESTDS
    TESTDS --> PROBS
    PROBS --> T1
    PROBS --> T2
    PROBS --> T3
    PROBS --> T4
    PROBS --> T5
    
    T1 --> REPORT
    T2 --> REPORT
    T3 --> REPORT
    T4 --> REPORT
    T5 --> REPORT
    
    REPORT --> OVERALL
    OVERALL --> VERDICT
    
    style T5 fill:#ffe6e6
    style REPORT fill:#e6f3ff
```

### Return Value Structure

Test 5 returns a dictionary (line 523-525) containing:

```python
{
    'uncertainty_ratio': float,           # incorrect_entropy / correct_entropy
    'high_certainty_accuracy': float,     # accuracy when entropy < threshold
    'mean_entropy_correct': float,        # average entropy for correct preds
    'mean_entropy_incorrect': float       # average entropy for incorrect preds
}
```

These metrics contribute to the overall evaluation score calculated at [Evaluate_model.py:669-701](../Evaluate_model.py#L669-L701).

**Sources:** [Evaluate_model.py:661-665](../Evaluate_model.py#L661-L665), [Evaluate_model.py:523-525](../Evaluate_model.py#L523-L525), [Evaluate_model.py:669-701](../Evaluate_model.py#L669-L701)

---

## Theoretical Justification

### Why Entropy Over Maximum Probability?

| Approach | Formula | Sensitivity |
|----------|---------|-------------|
| **Max Probability** (Test 3) | `max(p)` | Only captures dominant class |
| **Entropy** (Test 5) | `-Σ p*log(p)` | Captures full distribution shape |

**Example Comparison:**

| Prediction | Max Prob | Entropy | Interpretation |
|------------|----------|---------|----------------|
| `[0.95, 0.05, ...]` | 0.95 | 0.20 | High confidence, low uncertainty |
| `[0.50, 0.30, 0.20]` | 0.50 | 1.03 | Low confidence, high uncertainty |
| `[0.90, 0.08, 0.02]` | 0.90 | 0.42 | Moderate confidence, but entropy reveals some doubt |

Entropy detects cases where probability mass is distributed across multiple classes, indicating the model sees multiple plausible explanations—useful for identifying ambiguous defect patterns.

**Sources:** [Evaluate_model.py:451-452](../Evaluate_model.py#L451-L452)

---

## Configuration Parameters

Test 5 uses one primary configuration constant from the `Config` class:

```mermaid
graph LR
    CONFIG["Config class<br/>line 21-36"]
    THRESHOLD["ENTROPY_THRESHOLD<br/>= 0.5"]
    USAGE1["High-certainty mask<br/>line 463"]
    USAGE2["Visualization line<br/>line 502"]
    USAGE3["Console output<br/>line 466"]
    
    CONFIG --> THRESHOLD
    THRESHOLD --> USAGE1
    THRESHOLD --> USAGE2
    THRESHOLD --> USAGE3
    
    style CONFIG fill:#f9f9f9
    style THRESHOLD fill:#ffe6e6
```

**Threshold Tuning Guidance:**

- **Lower threshold (0.3)**: Stricter definition of "high certainty", fewer predictions qualify
- **Higher threshold (0.7)**: More lenient, but may include genuinely uncertain predictions
- **Current value (0.5)**: Balanced, roughly corresponding to 60-70% probability for top class

**Sources:** [Evaluate_model.py:36](../Evaluate_model.py#L36), [Evaluate_model.py:463](../Evaluate_model.py#L463), [Evaluate_model.py:502](../Evaluate_model.py#L502)

---

## Relationship to Other Tests

```mermaid
graph TB
    subgraph "Uncertainty Quantification Tests"
        T3["TEST 3: Confidence<br/>Max probability<br/>Threshold: 0.7-0.9"]
        T5["TEST 5: Entropy<br/>Full distribution<br/>Threshold: 0.5"]
    end
    
    subgraph "Complementary Insights"
        CONF["High max prob<br/>Low entropy"]
        UNC["Low max prob<br/>High entropy"]
        SPLIT["High max prob<br/>High entropy"]
    end
    
    subgraph "Interpretation"
        CONF_I["Confident, unimodal<br/>Likely correct"]
        UNC_I["Uncertain, diffuse<br/>Likely incorrect"]
        SPLIT_I["Bimodal distribution<br/>Between-class ambiguity"]
    end
    
    T3 --> CONF
    T5 --> CONF
    T3 --> UNC
    T5 --> UNC
    T3 --> SPLIT
    T5 --> SPLIT
    
    CONF --> CONF_I
    UNC --> UNC_I
    SPLIT --> SPLIT_I
    
    style T5 fill:#ffe6e6
    style SPLIT_I fill:#fff4e6
```

**Test Synergies:**

- **Test 1 (Confusion Matrix)**: Entropy analysis can identify which class pairs cause high-uncertainty confusions
- **Test 3 (Confidence)**: Entropy provides complementary view—can catch overconfident but uncertain predictions
- **Test 4 (Robustness)**: Expected behavior is increased entropy under noise perturbation

**Sources:** [Evaluate_model.py:295-390](../Evaluate_model.py#L295-L390) (Test 3), [Evaluate_model.py:438-525](../Evaluate_model.py#L438-L525) (Test 5)

---

## Output Artifacts Summary

Test 5 produces the following files in `OUTPUT_DIR`:

| Artifact | Path | Content |
|----------|------|---------|
| **Visualization** | `test5_entropy_analysis.png` | 2-panel figure: entropy distributions + per-class bars |
| **Metrics** | `validation_report.json` | Contains `test5_entropy_analysis` key with metrics dict |
| **Console Log** | Terminal output | Formatted statistical summary with interpretation |

The PNG file is saved at [Evaluate_model.py:523](../Evaluate_model.py#L523) and referenced in Diagram 4 of the high-level system overview.

**Sources:** [Evaluate_model.py:523](../Evaluate_model.py#L523), [Evaluate_model.py:669-701](../Evaluate_model.py#L669-L701)

# Alternative Training Approaches




## Purpose and Scope

This document describes experimental training methodologies developed prior to the current production system. These approaches explored different strategies for handling extreme class imbalance and defect classification in wafer imagery. While not currently used in production, they informed the architectural decisions in the final system.

For information about the current production training approach, see [Core Training System](./Core_Training_System.md#2). For evaluation methodology applied to these models, see [Model Evaluation Framework](./Model_Evaluation_Framework.md#4).

---

## Overview

The repository contains two alternative training implementations in `Previous_training_scripts/`:

| Script | Approach | Key Innovation | Status |
|--------|----------|----------------|--------|
| `train2.py` | Two-Phase Training | ClassBalancedLoss + oversampling | Archived |
| `train1.py` | CNN-SVM Ensemble | Hybrid deep learning + classical ML | Archived |

Both approaches were designed to address the severe class imbalance in the wafer defect dataset, particularly the scarcity of "good" (defect-free) samples. The evolution from these methods to the current progressive resizing + FocalLoss approach represents a shift toward curriculum learning and better loss functions rather than synthetic data generation or ensemble architectures.

**Sources:** [Previous_training_scripts/train2.py:1-535](../Previous_training_scripts/train2.py#L1-L535), [Previous_training_scripts/train1.py:1-220](../Previous_training_scripts/train1.py#L1-L220)

---

## System Architecture Comparison

```mermaid
graph TB
    subgraph train2["train2.py: Two-Phase Training"]
        T2A["ClassBalancedLoss<br/>β=0.9999"]
        T2B["create_balanced_dataset()<br/>Oversample to 500/class"]
        T2C["SyntheticDataGenerator<br/>Heavy augmentation"]
        T2D["two_phase_training()<br/>Phase 1: LR=1e-3<br/>Phase 2: LR=1e-5"]
        T2E["find_optimal_threshold()<br/>Per-class thresholds"]
        
        T2B --> T2C
        T2C --> T2D
        T2A --> T2D
        T2D --> T2E
    end
    
    subgraph train1["train1.py: CNN-SVM Ensemble"]
        T1A["build_cnn()<br/>EfficientNetB0 frozen"]
        T1B["Feature Extractor<br/>layers[-2].output"]
        T1C["SVM RBF Kernel<br/>C=10.0, balanced"]
        T1D["Weighted Voting<br/>0.4*CNN + 0.6*SVM"]
        
        T1A --> T1B
        T1B --> T1C
        T1B --> T1D
        T1C --> T1D
    end
    
    subgraph current["Current: kaggle-notebook.ipynb"]
        CA["MobileNetV2 α=0.75"]
        CB["SEBlock attention"]
        CC["FocalLoss γ=1.5"]
        CD["Progressive Resizing<br/>128→160→224"]
        
        CA --> CB
        CB --> CC
        CC --> CD
    end
    
    train2 -.evolved to.-> current
    train1 -.evolved to.-> current
```

**Diagram: Architectural Evolution**

The diagram shows how both alternative approaches contributed insights to the current system: `train2.py` demonstrated the effectiveness of specialized loss functions for imbalanced data, while `train1.py` validated the need for robust feature extraction but revealed that pure neural approaches with proper loss functions outperform hybrid ensembles.

**Sources:** [Previous_training_scripts/train2.py:51-95](../Previous_training_scripts/train2.py#L51-L95), [Previous_training_scripts/train1.py:82-96](../Previous_training_scripts/train1.py#L82-L96)

---

## Two-Phase Training with Class Balancing (train2.py)

### Architecture Overview

The `train2.py` approach tackles extreme class imbalance through a three-pronged strategy:

1. **Synthetic oversampling** to balance training data
2. **Class-Balanced Loss** (CVPR 2019) based on effective number of samples
3. **Two-phase training**: balanced dataset → full dataset

**Configuration Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `IMG_SIZE` | 224 | Input resolution |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS_PHASE1` | 30 | Balanced training epochs |
| `EPOCHS_PHASE2` | 20 | Fine-tuning epochs |
| Target samples/class | 500 | Oversampling target |

**Sources:** [Previous_training_scripts/train2.py:7-16](../Previous_training_scripts/train2.py#L7-L16)

---

### Class-Balanced Loss Implementation

```mermaid
graph LR
    A["samples_per_class<br/>[n_1, n_2, ..., n_k]"]
    B["effective_num<br/>(1-β^n)/(1-β)"]
    C["weights<br/>1/effective_num"]
    D["Normalized weights<br/>sum = k classes"]
    E["Loss calculation<br/>weighted CE or Focal"]
    
    A --> B
    B --> C
    C --> D
    D --> E
    
    F["β=0.9999<br/>High sensitivity"] --> B
```

**Diagram: ClassBalancedLoss Computation Pipeline**

The `ClassBalancedLoss` class implements the CVPR 2019 paper's approach for handling imbalanced datasets. The loss reweights each sample based on the effective number of samples in its class, where β controls the degree of reweighting.

**Key Implementation Details:**

- **Formula**: `CB(p, y) = (1-β)/(1-β^n_y) * L(p, y)`
- **β=0.9999**: High value makes weights closer to inverse frequency
- **Loss types**: Supports both `"crossentropy"` and `"focal"` (γ=2.0)
- **Normalization**: Weights sum to number of classes for stability

**Class Definition:**

The `ClassBalancedLoss` class [Previous_training_scripts/train2.py:51-95](../Previous_training_scripts/train2.py#L51-L95) extends `tf.keras.losses.Loss` with the following key methods:

- `__init__(samples_per_class, beta, loss_type, gamma)`: Computes and stores class weights
- `call(y_true, y_pred)`: Applies weights to cross-entropy or focal loss

**Weight Calculation Example:**

For a class with 5 samples (like "good" wafers) and β=0.9999:
- `effective_num = (1 - 0.9999^5) / (1 - 0.9999) ≈ 5.0`
- For a class with 1000 samples: `effective_num ≈ 632.1`
- This gives the minority class ~126x higher weight after normalization

**Sources:** [Previous_training_scripts/train2.py:51-95](../Previous_training_scripts/train2.py#L51-L95)

---

### Data Balancing Pipeline

```mermaid
flowchart TD
    A["analyze_class_distribution()<br/>Count samples per class"]
    B["Identify minority classes<br/>Min: 5 'good' samples<br/>Max: ~1000s defect samples"]
    C["create_balanced_dataset()<br/>target=500/class"]
    D["Calculate repeat factors<br/>good: 100x<br/>majority: 1x"]
    E["Expand file lists<br/>repeat images N times"]
    F["SyntheticDataGenerator<br/>Heavy augmentation per epoch"]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    
    G["create_augmentation_pipeline()<br/>is_minority_class flag"] --> F
```

**Diagram: Oversampling and Augmentation Flow**

**Balancing Strategy:**

The `create_balanced_dataset()` function [Previous_training_scripts/train2.py:100-141](../Previous_training_scripts/train2.py#L100-L141) creates a training set where all classes have approximately equal representation:

1. **Analysis Phase**: `analyze_class_distribution()` [Previous_training_scripts/train2.py:21-46](../Previous_training_scripts/train2.py#L21-L46) computes class counts
2. **Repeat Factor Calculation**: Each class is repeated `max(1, target / n_samples)` times
3. **File List Expansion**: Image paths are duplicated in the dataset

**Augmentation Differentiation:**

The `create_augmentation_pipeline()` function [Previous_training_scripts/train2.py:146-169](../Previous_training_scripts/train2.py#L146-L169) applies different augmentation strategies based on class size:

| Augmentation | Minority Classes | Majority Classes |
|--------------|------------------|------------------|
| RandomFlip | horizontal_and_vertical | horizontal only |
| RandomRotation | 0.5 (180°) | 0.25 (90°) |
| RandomZoom | 0.2 | N/A |
| RandomTranslation | 0.2, 0.2 | N/A |
| RandomBrightness | 0.3 | 0.1 |
| RandomContrast | 0.3 | 0.1 |

**Rationale**: Minority classes need more aggressive augmentation to create diverse synthetic samples, while majority classes preserve defect patterns with conservative transforms.

**Sources:** [Previous_training_scripts/train2.py:21-46](../Previous_training_scripts/train2.py#L21-L46), [Previous_training_scripts/train2.py:100-141](../Previous_training_scripts/train2.py#L100-L141), [Previous_training_scripts/train2.py:146-169](../Previous_training_scripts/train2.py#L146-L169)

---

### Two-Phase Training Strategy

```mermaid
stateDiagram-v2
    [*] --> Phase1
    
    state Phase1 {
        [*] --> BalancedData: "Oversampled to 500/class"
        BalancedData --> CBLoss: "ClassBalancedLoss β=0.9999"
        CBLoss --> AdamW1: "LR=1e-3, weight_decay=1e-4"
        AdamW1 --> Callbacks1: "30 epochs max"
        Callbacks1 --> BestModel1: "Monitor val_f1"
    }
    
    Phase1 --> Phase2: "Load best_balanced.keras"
    
    state Phase2 {
        [*] --> FullData: "Original imbalanced distribution"
        FullData --> CBLoss2: "Same ClassBalancedLoss"
        CBLoss2 --> AdamW2: "LR=1e-5 (100x lower)"
        AdamW2 --> Callbacks2: "20 epochs max"
        Callbacks2 --> BestModel2: "Monitor val_f1"
    }
    
    Phase2 --> [*]: "best_finetuned.keras"
```

**Diagram: Two-Phase Training Workflow**

**Phase 1: Balanced Training**

The first phase [Previous_training_scripts/train2.py:192-243](../Previous_training_scripts/train2.py#L192-L243) trains on the artificially balanced dataset:

- **Objective**: Learn to recognize all classes equally
- **Learning Rate**: 1e-3 (standard initial rate)
- **Callbacks**:
  - `EarlyStopping`: patience=10, monitor `val_f1`
  - `ReduceLROnPlateau`: factor=0.5, patience=5
  - `ModelCheckpoint`: saves `best_balanced.keras`
- **Metrics**: Accuracy, Precision, Recall, F1 (weighted)

**Phase 2: Fine-Tuning**

The second phase [Previous_training_scripts/train2.py:245-289](../Previous_training_scripts/train2.py#L245-L289) adapts to real-world class distribution:

- **Objective**: Adjust decision boundaries to actual class frequencies
- **Learning Rate**: 1e-5 (100x lower for gentle adaptation)
- **Dataset**: Original imbalanced training data
- **Callbacks**:
  - `EarlyStopping`: patience=15 (longer for stability)
  - `ReduceLROnPlateau`: factor=0.3, patience=7
  - `ModelCheckpoint`: saves `best_finetuned.keras`

**Key Function: `two_phase_training()`**

The orchestrator function [Previous_training_scripts/train2.py:186-291](../Previous_training_scripts/train2.py#L186-L291) accepts:
- `model`: Keras model to train
- `train_ds_balanced`: Oversampled dataset
- `train_ds_full`: Original imbalanced dataset
- `val_ds`: Validation dataset
- `class_counts`: Dictionary of class frequencies

**Sources:** [Previous_training_scripts/train2.py:174-291](../Previous_training_scripts/train2.py#L174-L291)

---

### Optimal Threshold Finding

```mermaid
graph TD
    A["find_optimal_threshold()<br/>Per-class threshold tuning"]
    B["Collect predictions<br/>y_scores, y_true"]
    C["For each class i"]
    D["Binary problem:<br/>class i vs rest"]
    E["Search thresholds<br/>0.1 to 0.9, step 0.05"]
    F["Calculate F1 score<br/>for each threshold"]
    G["Select threshold<br/>with max F1"]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    
    H["optimal_thresholds<br/>dict[class_name, threshold]"] --> I["evaluate_with_proper_metrics()<br/>Apply custom thresholds"]
```

**Diagram: Threshold Optimization Process**

**Motivation:**

Standard classification uses a fixed 0.5 threshold for all classes. For imbalanced datasets, optimal thresholds vary by class. A minority class might require a lower threshold to improve recall.

**Algorithm:**

The `find_optimal_threshold()` function [Previous_training_scripts/train2.py:296-345](../Previous_training_scripts/train2.py#L296-L345) implements a grid search:

1. **Prediction Collection**: Gather all validation predictions
2. **Per-Class Search**: For each class, treat as binary classification problem
3. **Grid Search**: Test thresholds from 0.1 to 0.9 in steps of 0.05
4. **F1 Maximization**: Select threshold that maximizes F1 score
5. **Return Dictionary**: Maps class names to optimal thresholds

**Example Output:**

```
good                : threshold=0.30, F1=0.847
Center             : threshold=0.55, F1=0.921
Donut              : threshold=0.50, F1=0.889
```

**Application in Inference:**

The `evaluate_with_proper_metrics()` function [Previous_training_scripts/train2.py:350-435](../Previous_training_scripts/train2.py#L350-L435) applies these thresholds:

```python
if score[i] >= optimal_thresholds[class_name] and score[i] > max_score:
    pred_class = i
```

This replaces the standard `np.argmax()` with threshold-aware prediction.

**Sources:** [Previous_training_scripts/train2.py:296-345](../Previous_training_scripts/train2.py#L296-L345), [Previous_training_scripts/train2.py:350-435](../Previous_training_scripts/train2.py#L350-L435)

---

### SyntheticDataGenerator

The `SyntheticDataGenerator` class [Previous_training_scripts/train2.py:440-498](../Previous_training_scripts/train2.py#L440-L498) implements SMOTE-style augmentation for images:

**Architecture:**

```mermaid
classDiagram
    class SyntheticDataGenerator {
        +image_paths: list
        +labels: list
        +batch_size: int
        +target_samples: int
        +repeat_factor: int
        +expanded_paths: list
        +expanded_labels: list
        +augment: Sequential
        __init__(paths, labels, batch_size, target)
        __len__() int
        __getitem__(idx) tuple
    }
    
    class Sequential {
        +RandomFlip
        +RandomRotation(0.5)
        +RandomZoom(0.3)
        +RandomTranslation(0.3, 0.3)
        +RandomBrightness(0.4)
        +RandomContrast(0.4)
    }
    
    SyntheticDataGenerator --> Sequential : uses
```

**Key Features:**

1. **Expansion**: Repeats each sample `repeat_factor` times in memory
2. **Per-Epoch Variation**: Applies different augmentation each epoch
3. **Heavy Augmentation**: More aggressive than standard pipelines
4. **Memory Efficient**: Generates augmented images on-the-fly during training

**Usage Pattern:**

```python