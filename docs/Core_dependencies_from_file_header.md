import tensorflow as tf                    # Model training, layers
import numpy as np                         # Array operations
from collections import Counter            # Class counting
from sklearn.utils.class_weight import compute_class_weight  # Class weight computation
from sklearn.metrics import (
    classification_report,                # Detailed metrics
    confusion_matrix,                     # Misclassification analysis
    precision_recall_fscore_support       # Per-class metrics
)
```

The script requires:
- TensorFlow 2.x with Keras API
- scikit-learn for metric computation
- NumPy for numerical operations
- Access to preprocessed grayscale images organized by class

**Sources:** [Previous_training_scripts/train2.py:1-5](), [Previous_training_scripts/train2.py:354](), [Previous_training_scripts/train2.py:401]()

---

## Execution Workflow

The main execution flow follows a linear pipeline:

```mermaid
sequenceDiagram
    participant Main as main()
    participant Analyze as analyze_class_distribution()
    participant Balance as create_balanced_dataset()
    participant Build as build_optimized_model()
    participant Train as two_phase_training()
    participant Thresh as find_optimal_threshold()
    participant Eval as evaluate_with_proper_metrics()
    
    Main->>Analyze: Analyze TRAIN_DIR
    Analyze-->>Main: class_counts dict
    
    Main->>Balance: Create balanced dataset (500/class)
    Balance-->>Main: balanced_files, balanced_labels, class_names
    
    Main->>Build: Build model(n_classes)
    Build-->>Main: model, base
    
    Note over Main: Create tf.data pipelines<br/>train_ds_balanced, train_ds_full, val_ds
    
    Main->>Train: two_phase_training()<br/>(model, datasets, class_counts)
    Train->>Train: Phase 1: Balanced (30 epochs)
    Train->>Train: Phase 2: Fine-tune (20 epochs)
    Train-->>Main: Trained model
    
    Main->>Thresh: find_optimal_threshold()<br/>(model, val_ds, class_names)
    Thresh-->>Main: optimal_thresholds dict
    
    Main->>Eval: evaluate_with_proper_metrics()<br/>(model, test_ds, thresholds)
    Eval-->>Main: metrics dict
    
    Main->>Main: Save model to<br/>wafer_classifier_imbalanced.keras
```

The `main()` function [Previous_training_scripts/train2.py:503-531]() orchestrates the complete pipeline, though the actual dataset creation code (step 4 in the function) is omitted from the file, indicated by the comment `# ... (dataset creation code)` at line 517.

**Sources:** [Previous_training_scripts/train2.py:503-534]()