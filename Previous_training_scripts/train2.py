import tensorflow as tf
import numpy as np
import os
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# ===========================
# CONFIGURATION
# ===========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 30  # Balanced training
EPOCHS_PHASE2 = 20  # Fine-tuning on full data

TRAIN_DIR = "/kaggle/input/data1423/Segregated_defects_grayscale/train"
TEST_DIR = "/kaggle/input/data1423/Segregated_defects_grayscale/test"

# ===========================
# STEP 1: ANALYZE CLASS DISTRIBUTION
# ===========================
def analyze_class_distribution(directory):
    """Analyze the class distribution in the dataset"""
    class_counts = {}
    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            class_counts[class_name] = count
    
    total = sum(class_counts.values())
    print("\n📊 CLASS DISTRIBUTION ANALYSIS:")
    print("=" * 60)
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1]):
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 2)
        print(f"{class_name:25s}: {count:5d} ({percentage:5.2f}%) {bar}")
    print("=" * 60)
    print(f"Total images: {total}")
    
    # Identify minority classes
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"\nMost imbalanced ratio: {max_count/min_count:.1f}:1")
    
    return class_counts

# ===========================
# STEP 2: CLASS-BALANCED LOSS (CVPR 2019)
# ===========================
class ClassBalancedLoss(tf.keras.losses.Loss):
    """
    Class-Balanced Loss Based on Effective Number of Samples
    Paper: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    
    Formula: CB(p, y) = (1-β)/(1-β^n_y) * L(p, y)
    where n_y is the number of samples in class y
    """
    def __init__(self, samples_per_class, beta=0.9999, loss_type="focal", gamma=2.0, name="class_balanced_loss"):
        super().__init__(name=name)
        self.samples_per_class = tf.constant(samples_per_class, dtype=tf.float32)
        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma
        
        # Calculate effective number of samples: (1-β^n)/(1-β)
        effective_num = (1.0 - tf.pow(beta, self.samples_per_class)) / (1.0 - beta)
        
        # Weights are inversely proportional to effective number
        # Normalize so that sum of weights = number of classes
        weights = 1.0 / effective_num
        self.weights = weights / tf.reduce_sum(weights) * len(samples_per_class)
        
        print(f"\n📊 Class-Balanced Weights (β={beta}):")
        for i, (n, w) in enumerate(zip(samples_per_class, self.weights.numpy())):
            print(f"  Class {i}: n={n:5d}, weight={w:.4f}")
    
    def call(self, y_true, y_pred):
        # Apply class weights
        weights = tf.reduce_sum(self.weights * y_true, axis=-1)
        
        if self.loss_type == "crossentropy":
            # Class-Balanced Cross-Entropy
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            return tf.reduce_mean(weights * ce)
        
        elif self.loss_type == "focal":
            # Class-Balanced Focal Loss
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
            focal_weight = tf.pow(1.0 - p_t, self.gamma)
            return tf.reduce_mean(weights * focal_weight * ce)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

# ===========================
# STEP 3: CREATE BALANCED DATASET VIA OVERSAMPLING
# ===========================
def create_balanced_dataset(directory, target_samples_per_class=500):
    """
    Create a balanced dataset by oversampling minority classes
    
    For the 'good' class with only 5 images:
    - Apply aggressive augmentation
    - Repeat each image ~100 times with different augmentations
    """
    
    class_paths = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, f) for f in os.listdir(class_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            class_paths[class_name] = images
    
    # Calculate how many times to repeat each class
    class_repeat_factors = {}
    for class_name, paths in class_paths.items():
        n_samples = len(paths)
        repeat_factor = max(1, target_samples_per_class // n_samples)
        class_repeat_factors[class_name] = repeat_factor
        print(f"{class_name}: {n_samples} samples → repeat {repeat_factor}x → ~{n_samples * repeat_factor} samples")
    
    # Create file list with repeats
    all_files = []
    all_labels = []
    class_names = sorted(class_paths.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name, paths in class_paths.items():
        repeat = class_repeat_factors[class_name]
        for _ in range(repeat):
            for path in paths:
                all_files.append(path)
                all_labels.append(class_to_idx[class_name])
    
    print(f"\n📊 Balanced dataset: {len(all_files)} total samples")
    print(f"Target per class: ~{target_samples_per_class}")
    
    return all_files, all_labels, class_names

# ===========================
# STEP 4: AGGRESSIVE AUGMENTATION FOR MINORITY CLASS
# ===========================
def create_augmentation_pipeline(is_minority_class=False):
    """
    Create augmentation pipeline
    Minority classes get MORE aggressive augmentation
    """
    
    if is_minority_class:
        # AGGRESSIVE augmentation for clean wafers (minority class)
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.5),  # 180 degrees
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.2, 0.2),
            tf.keras.layers.RandomBrightness(0.3),
            tf.keras.layers.RandomContrast(0.3),
        ])
    else:
        # Conservative augmentation for majority classes
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.25),  # 90 degrees
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])

# ===========================
# STEP 5: TWO-PHASE TRAINING STRATEGY
# ===========================
"""
PHASE 1: Train on BALANCED dataset
- Oversample minority classes (especially 'good')
- Use Class-Balanced Loss
- This teaches the model to recognize ALL classes

PHASE 2: Fine-tune on FULL dataset
- Use original imbalanced distribution
- Lower learning rate
- This adapts to real-world distribution
"""

def two_phase_training(model, train_ds_balanced, train_ds_full, val_ds, class_counts):
    """
    Two-phase training for extreme imbalance
    """
    samples_per_class = [class_counts.get(name, 1) for name in sorted(class_counts.keys())]
    
    # ========== PHASE 1: Balanced Training ==========
    print("\n" + "="*70)
    print("PHASE 1: Training on BALANCED dataset (oversampled)")
    print("="*70)
    
    # Use Class-Balanced Focal Loss
    cb_loss = ClassBalancedLoss(
        samples_per_class=samples_per_class,
        beta=0.9999,  # Higher = closer to inverse frequency
        loss_type="focal",
        gamma=2.0
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=cb_loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(average='weighted', name='f1')
        ]
    )
    
    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_balanced.keras',
            monitor='val_f1',
            save_best_only=True,
            mode='max'
        )
    ]
    
    history1 = model.fit(
        train_ds_balanced,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks_phase1
    )
    
    # ========== PHASE 2: Fine-tuning on Full Dataset ==========
    print("\n" + "="*70)
    print("PHASE 2: Fine-tuning on FULL dataset (imbalanced)")
    print("="*70)
    
    # Use lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss=cb_loss,  # Still use class-balanced loss
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(average='weighted', name='f1')
        ]
    )
    
    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1',
            patience=15,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1',
            factor=0.3,
            patience=7,
            min_lr=1e-8,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_finetuned.keras',
            monitor='val_f1',
            save_best_only=True,
            mode='max'
        )
    ]
    
    history2 = model.fit(
        train_ds_full,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks_phase2
    )
    
    return model

# ===========================
# STEP 6: THRESHOLD ADJUSTMENT FOR INFERENCE
# ===========================
def find_optimal_threshold(model, val_ds, class_names):
    """
    Find optimal classification threshold for each class
    Especially important for the minority 'good' class
    """
    print("\n🔍 Finding optimal thresholds...")
    
    # Collect predictions and true labels
    y_true = []
    y_scores = []
    
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_scores.extend(preds)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Find threshold that maximizes F1 for each class
    optimal_thresholds = {}
    
    for i, class_name in enumerate(class_names):
        # Binary classification: class i vs rest
        y_true_binary = (np.argmax(y_true, axis=1) == i).astype(int)
        y_score_class = y_scores[:, i]
        
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred_binary = (y_score_class >= thresh).astype(int)
            
            # Calculate F1
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        optimal_thresholds[class_name] = best_thresh
        print(f"  {class_name:20s}: threshold={best_thresh:.2f}, F1={best_f1:.3f}")
    
    return optimal_thresholds

# ===========================
# STEP 7: PROPER EVALUATION (NOT ACCURACY!)
# ===========================
def evaluate_with_proper_metrics(model, test_ds, class_names, optimal_thresholds=None):
    """
    Evaluate using metrics that matter for imbalanced data
    """
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    
    print("\n" + "="*70)
    print("EVALUATION WITH PROPER METRICS")
    print("="*70)
    
    # Collect predictions
    y_true = []
    y_pred = []
    y_scores = []
    
    for images, labels in test_ds:
        scores = model.predict(images, verbose=0)
        y_scores.extend(scores)
        
        if optimal_thresholds:
            # Use custom thresholds
            batch_preds = []
            for score in scores:
                pred_class = None
                max_score = 0
                for i, class_name in enumerate(class_names):
                    if score[i] >= optimal_thresholds[class_name] and score[i] > max_score:
                        max_score = score[i]
                        pred_class = i
                if pred_class is None:
                    pred_class = np.argmax(score)  # Fallback
                batch_preds.append(pred_class)
            y_pred.extend(batch_preds)
        else:
            y_pred.extend(np.argmax(scores, axis=1))
        
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    
    # Per-class metrics
    print("\n📊 Per-Class Performance:")
    print("-" * 70)
    print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 70)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {int(support[i]):<10}")
    
    print("-" * 70)
    print(f"{'Weighted Avg':<25} {np.average(precision, weights=support):<10.3f} "
          f"{np.average(recall, weights=support):<10.3f} "
          f"{np.average(f1, weights=support):<10.3f}")
    
    # Confusion Matrix
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Check for 'good' class specifically
    good_idx = class_names.index('good') if 'good' in class_names else None
    if good_idx is not None:
        good_recall = recall[good_idx]
        good_precision = precision[good_idx]
        print(f"\n🎯 'Good' (clean wafer) class performance:")
        print(f"   Precision: {good_precision:.3f}")
        print(f"   Recall: {good_recall:.3f}")
        print(f"   F1-Score: {f1[good_idx]:.3f}")
        
        if good_recall < 0.5:
            print("   ⚠️ WARNING: Low recall for 'good' class - model is biased toward defects!")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }

# ===========================
# STEP 8: SMOTE-STYLE SYNTHETIC DATA GENERATION
# ===========================
class SyntheticDataGenerator(tf.keras.utils.Sequence):
    """
    Generate synthetic samples for minority classes using heavy augmentation
    Similar to SMOTE but for images
    """
    def __init__(self, image_paths, labels, batch_size, target_samples):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_samples = target_samples
        
        # Calculate how many times to repeat each sample
        self.n_original = len(image_paths)
        self.repeat_factor = max(1, target_samples // self.n_original)
        
        # Create expanded dataset
        self.expanded_paths = []
        self.expanded_labels = []
        
        for _ in range(self.repeat_factor):
            self.expanded_paths.extend(image_paths)
            self.expanded_labels.extend(labels)
        
        # Shuffle
        indices = np.random.permutation(len(self.expanded_paths))
        self.expanded_paths = [self.expanded_paths[i] for i in indices]
        self.expanded_labels = [self.expanded_labels[i] for i in indices]
        
        self.augment = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.5),
            tf.keras.layers.RandomZoom(0.3),
            tf.keras.layers.RandomTranslation(0.3, 0.3),
            tf.keras.layers.RandomBrightness(0.4),
            tf.keras.layers.RandomContrast(0.4),
        ])
    
    def __len__(self):
        return len(self.expanded_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_paths = self.expanded_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.expanded_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_images = []
        for path in batch_paths:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=1, expand_animations=False)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img = tf.cast(img, tf.float32) / 255.0
            batch_images.append(img)
        
        batch_images = tf.stack(batch_images)
        batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=14)
        
        # Apply augmentation
        batch_images = self.augment(batch_images, training=True)
        
        return batch_images, batch_labels

# ===========================
# MAIN EXECUTION
# ===========================
def main():
    # 1. Analyze class distribution
    class_counts = analyze_class_distribution(TRAIN_DIR)
    
    # 2. Create balanced dataset
    balanced_files, balanced_labels, class_names = create_balanced_dataset(
        TRAIN_DIR, 
        target_samples_per_class=500
    )
    
    # 3. Build model
    model, base = build_optimized_model(len(class_names))
    
    # 4. Create datasets
    # ... (dataset creation code)
    
    # 5. Two-phase training
    model = two_phase_training(model, train_ds_balanced, train_ds_full, val_ds, class_counts)
    
    # 6. Find optimal thresholds
    optimal_thresholds = find_optimal_threshold(model, val_ds, class_names)
    
    # 7. Evaluate with proper metrics
    metrics = evaluate_with_proper_metrics(model, test_ds, class_names, optimal_thresholds)
    
    # 8. Save model
    model.save("wafer_classifier_imbalanced.keras")
    
    return model, metrics

if __name__ == "__main__":
    main()
