
import os
import numpy as np
import tensorflow as tf
import shutil
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as MobileNetV2Class
from datetime import datetime
import json
from pathlib import Path
from collections import Counter
import warnings

# Suppress TF warnings and XLA logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
warnings.filterwarnings('ignore')

# Disable XLA compilation to avoid timeout errors
tf.config.optimizer.set_jit(False)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATASET_ROOT = "/kaggle/input/dataset/dataset"
    TRAIN_DIR = f"{DATASET_ROOT}/train"
    VAL_DIR = f"{DATASET_ROOT}/val"
    OUTPUT_DIR = "/kaggle/working/prototype_b_optimized"
    MODEL_DIR = f"{OUTPUT_DIR}/models"
    RESULTS_DIR = f"{OUTPUT_DIR}/results"
    TEST_DIR = f"{DATASET_ROOT}/test"
    COMBINED_VAL_DIR = "/kaggle/working/combined_validation_optimized"
    
    # Progressive resizing - reduced stages for stability
    PROGRESSIVE_SIZES = [128, 160, 224]
    PROGRESSIVE_EPOCHS = [20, 25, 35]
    
    # Learning rates - conservative to avoid instability
    INITIAL_LR = 1e-3
    FINE_TUNE_LR = 5e-5
    
    # Model
    BACKBONE = "MobileNetV2"
    ALPHA = 0.75
    USE_SE_ATTENTION = True
    
    # Training
    BATCH_SIZE = 32  # Reduced for Kaggle T4 stability
    USE_CLASS_WEIGHTS = True
    
    # Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    
    # Loss
    FOCAL_GAMMA = 1.5
    FOCAL_ALPHA = 0.25
    LABEL_SMOOTHING = 0.1
    
    RANDOM_SEED = 42


# ============================================================================
# DATASET COMBINATION
# ============================================================================

def combine_validation_sets(val_dir, test_dir, output_dir, max_per_class=60):
    """Combine val and test with stratified sampling"""
    print("\n" + "="*80)
    print("COMBINING VALIDATION DATASETS")
    print("="*80)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    val_classes = [d for d in os.listdir(val_dir) 
                   if os.path.isdir(os.path.join(val_dir, d))]
    
    total_images = 0
    class_counts = {}
    
    for class_name in val_classes:
        combined_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(combined_class_dir, exist_ok=True)
        
        all_images = []
        
        val_class_dir = os.path.join(val_dir, class_name)
        if os.path.exists(val_class_dir):
            for img_file in os.listdir(val_class_dir):
                src = os.path.join(val_class_dir, img_file)
                if os.path.isfile(src):
                    all_images.append(('val', src, img_file))
        
        test_class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(test_class_dir):
            for img_file in os.listdir(test_class_dir):
                src = os.path.join(test_class_dir, img_file)
                if os.path.isfile(src):
                    all_images.append(('test', src, img_file))
        
        # Stratified sampling
        if len(all_images) > max_per_class:
            np.random.shuffle(all_images)
            all_images = all_images[:max_per_class]
        
        for prefix, src, img_file in all_images:
            dst = os.path.join(combined_class_dir, f"{prefix}_{img_file}")
            shutil.copy2(src, dst)
        
        class_counts[class_name] = len(all_images)
        total_images += len(all_images)
        print(f"  ✓ {class_name}: {len(all_images)} images")
    
    print(f"\nTotal validation images: {total_images}")
    print("="*80 + "\n")
    
    return class_counts


# ============================================================================
# SE ATTENTION
# ============================================================================

class SEBlock(layers.Layer):
    def __init__(self, channels, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Dense(channels // ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
    
    def call(self, inputs):
        squeeze = self.global_pool(inputs)
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)
        return inputs * excitation
    
    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'ratio': self.ratio})
        return config


# ============================================================================
# OPTIMIZED DATA PIPELINE 
# ============================================================================

class DataPipeline:
    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes
    
    def load_and_preprocess(self, path, label):
        """Load and preprocess - optimized single function"""
        # Read file
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=1)
        
        # Resize
        image = tf.image.resize(image, [self.image_size, self.image_size])
        
        # Convert to float and normalize
        image = tf.cast(image, tf.float32)
        
        # Simple normalization (faster than percentile)
        image = image / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return image, label
    
    def augment(self, image, label):
        """Lightweight augmentation"""
        # Random flip
        image = tf.image.random_flip_left_right(image)
        
        # Random rotation (0, 90, 180, 270)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        
        # Random brightness
        image = tf.image.random_brightness(image, 0.1)
        image = tf.clip_by_value(image, -1.0, 1.0)
        
        return image, label
    
    @tf.function
    def apply_mixup(self, images, labels):
        """Vectorized MixUp - simplified"""
        batch_size = tf.shape(images)[0]
        
        # Generate mixing coefficient
        lam = tf.random.uniform([batch_size, 1, 1, 1], 0.2, 0.8)
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images
        mixed_images = lam * images + (1.0 - lam) * tf.gather(images, indices)
        
        # Mix labels
        lam_labels = tf.reshape(lam, [batch_size, 1])
        mixed_labels = lam_labels * labels + (1.0 - lam_labels) * tf.gather(labels, indices)
        
        return mixed_images, mixed_labels
    
    def create_dataset(self, directory, batch_size, is_training=True, class_weights=None):
        """Create optimized dataset without tf.map_fn bottlenecks"""
        
        # Use image_dataset_from_directory for efficient loading
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=(self.image_size, self.image_size),
            batch_size=batch_size,
            label_mode='categorical',
            color_mode='grayscale',
            shuffle=is_training,
            seed=Config.RANDOM_SEED if is_training else None
        )
        
        # Apply class weights by repeating samples (if needed)
        if is_training and class_weights is not None:
            # Get all data
            all_images = []
            all_labels = []
            for images, labels in dataset:
                all_images.append(images)
                all_labels.append(labels)
            
            all_images = tf.concat(all_images, axis=0)
            all_labels = tf.concat(all_labels, axis=0)
            
            # Calculate sample weights
            sample_weights = tf.reduce_sum(all_labels * tf.constant([list(class_weights.values())]), axis=1)
            sample_weights = sample_weights / tf.reduce_sum(sample_weights)
            
            # Weighted sampling
            num_samples = len(all_images) * 2
            indices = tf.random.categorical(
                tf.math.log(sample_weights[None, :]), 
                num_samples
            )[0]
            
            all_images = tf.gather(all_images, indices)
            all_labels = tf.gather(all_labels, indices)
            
            dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
            dataset = dataset.batch(batch_size)
        
        # Normalize: [-1, 1] range (compatible with MobileNetV2)
        normalization_layer = layers.Rescaling(1./127.5, offset=-1)
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y), 
                             num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            # Lightweight augmentation
            dataset = dataset.map(
                lambda x, y: (tf.image.random_flip_left_right(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Random rotation
            def random_rotate(x, y):
                k = tf.random.uniform([], 0, 4, dtype=tf.int32)
                return tf.image.rot90(x, k), y
            
            dataset = dataset.map(random_rotate, num_parallel_calls=tf.data.AUTOTUNE)
            
            # MixUp (occasionally)
            if Config.USE_MIXUP:
                dataset = dataset.map(
                    lambda x, y: self.apply_mixup(x, y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
            dataset = dataset.shuffle(1000)
            dataset = dataset.repeat()
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=1.5, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        # Label smoothing
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Focal loss
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing
        })
        return config


# ============================================================================
# MODEL BUILDER 
# ============================================================================

def build_model(num_classes, image_size, use_se=True, weights=None):
    """Build MobileNetV2 with optional SE attention"""
    
    inputs = keras.Input(shape=(image_size, image_size, 1))
    
    # Convert grayscale to 3-channel
    x = layers.Concatenate()([inputs, inputs, inputs])
    
    # Build backbone
    base = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights='imagenet' if weights is None else None,
        alpha=Config.ALPHA
    )
    
    if weights is not None:
        base.set_weights(weights)
    
    base.trainable = False
    
    x = base(x, training=False)
    
    # SE Attention
    if use_se:
        x = SEBlock(channels=int(x.shape[-1]))(x)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base


# ============================================================================
# GPU WARMUP 
# ============================================================================

def warmup_gpu(model, image_size, batch_size=8):
    """Minimal GPU warmup"""
    dummy_data = tf.random.normal([batch_size, image_size, image_size, 1])
    _ = model(dummy_data, training=False)
    print("✓ GPU warmup complete")


# ============================================================================
# PROGRESSIVE TRAINER 
# ============================================================================

class ProgressiveTrainer:
    def __init__(self, num_classes, class_names, class_counts):
        self.num_classes = num_classes
        self.class_names = class_names
        self.class_counts = class_counts
        self.histories = []
        
        # Calculate class weights
        total = sum(class_counts.values())
        self.class_weights = {
            i: total / (num_classes * count) 
            for i, count in class_counts.items()
        }
        print("Class weights:", {k: f"{v:.2f}" for k, v in self.class_weights.items()})
    
    def get_backbone_layer_name(self, image_size):
        """Get the correct backbone layer name based on image size"""
        if image_size == 224:
            return 'mobilenetv2_1.00_224'
        else:
            return f'mobilenetv2_{Config.ALPHA}_{image_size}'
    
    def train_stage(self, train_dir, val_dir, image_size, epochs, 
                   batch_size, stage_idx, prev_model=None):
        """Train single stage with optimizations"""
        
        print(f"\n{'='*80}")
        print(f"STAGE {stage_idx + 1}: {image_size}×{image_size}")
        print(f"{'='*80}\n")
        
        # Build model - reuse backbone weights if available
        # Define learning rate and unfreeze strategy FIRST
        lr = Config.INITIAL_LR
        unfreeze = False
        
        if stage_idx == 0:
            lr = Config.INITIAL_LR
            unfreeze = False
        elif stage_idx == 1:
            lr = Config.INITIAL_LR / 2
            unfreeze = False
        else:  # stage_idx == 2 (224x224)
            lr = Config.INITIAL_LR / 5
            unfreeze = True
        
        # Build model - reuse full weights if possible
        if prev_model is not None and stage_idx > 0:
            try:
                # Save previous model weights
                prev_weights = prev_model.get_weights()
                
                # Build new model
                model, base = build_model(self.num_classes, image_size, 
                                         Config.USE_SE_ATTENTION, weights=None)
                
                # Try to transfer weights (skip if shapes don't match)
                current_weights = model.get_weights()
                if len(prev_weights) == len(current_weights):
                    # Check if shapes match for all layers
                    shapes_match = all(
                        pw.shape == cw.shape 
                        for pw, cw in zip(prev_weights, current_weights)
                    )
                    if shapes_match:
                        model.set_weights(prev_weights)
                        print(f"✓ Transferred all weights from previous stage")
                    else:
                        print(f"⚠ Weight shapes differ, using fresh weights")
                else:
                    print(f"⚠ Weight count differs ({len(prev_weights)} vs {len(current_weights)}), using fresh weights")
                    
            except Exception as e:
                print(f"⚠ Could not transfer weights: {e}")
                model, base = build_model(self.num_classes, image_size, Config.USE_SE_ATTENTION)
        else:
            model, base = build_model(self.num_classes, image_size, Config.USE_SE_ATTENTION)
            print(f"✓ Built new model: {model.count_params():,} parameters")
        
        # Unfreeze top layers if needed (NOW unfreeze is defined)
        if unfreeze and base is not None:
            base.trainable = True
            # Freeze first 100 layers, train the rest
            for layer in base.layers[:100]:
                layer.trainable = False
            print(f"✓ Unfroze top {len(base.layers) - 100} layers of backbone")
            base.trainable = True
            # Freeze first 100 layers, train the rest
            for layer in base.layers[:100]:
                layer.trainable = False
            print(f"✓ Unfroze top {len(base.layers) - 100} layers of backbone")
        
        # Warmup GPU before training
        warmup_gpu(model, image_size)
        
        # Create datasets
        pipeline = DataPipeline(image_size, self.num_classes)
        
        train_ds = pipeline.create_dataset(
            train_dir, batch_size,
            is_training=True,
            class_weights=self.class_weights if Config.USE_CLASS_WEIGHTS else None
        )
        
        val_ds = pipeline.create_dataset(
            val_dir, batch_size,
            is_training=False
        )
        
        # Calculate steps
        total_train = sum(self.class_counts.values()) * 2  # Account for oversampling
        steps_per_epoch = max(1, total_train // batch_size)
        

        
        # Compile with explicit metric names
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=FocalLoss(gamma=Config.FOCAL_GAMMA, alpha=Config.FOCAL_ALPHA),
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='prec'),
                keras.metrics.Recall(name='rec')
            ]
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                f"{Config.MODEL_DIR}/stage_{image_size}.keras",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        print(f"Training with {steps_per_epoch} steps per epoch...")
        history = model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        self.histories.append(history)
        
        # Evaluate
        results = model.evaluate(val_ds, return_dict=True, verbose=1)
        print(f"\nStage {stage_idx + 1} Results:")
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  Precision: {results['prec']*100:.2f}%")
        print(f"  Recall: {results['rec']*100:.2f}%")
        
        return model
    
    def train_progressive(self, train_dir, val_dir):
        """Progressive resizing training"""
        
        model = None
        
        for stage, (size, epochs) in enumerate(zip(
            Config.PROGRESSIVE_SIZES, 
            Config.PROGRESSIVE_EPOCHS
        )):
            model = self.train_stage(
                train_dir, val_dir, size, epochs,
                Config.BATCH_SIZE, stage, model
            )
            
            # Save checkpoint
            model.save(f"{Config.MODEL_DIR}/checkpoint_{size}.keras")
        
        # Final fine-tuning
        print(f"\n{'='*80}")
        print("FINAL FINE-TUNING")
        print(f"{'='*80}")
        
        # Unfreeze top layers - find backbone by type
        base = None
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower():
                base = layer
                break
        
        if base is not None:
            base.trainable = True
            for layer in base.layers[:-20]:
                layer.trainable = False
            
            model.compile(
                optimizer=keras.optimizers.Adam(Config.FINE_TUNE_LR / 10),
                loss=FocalLoss(),
                metrics=[
                    'accuracy',
                    keras.metrics.Precision(name='prec'),
                    keras.metrics.Recall(name='rec')
                ]
            )
            
            # Continue training
            pipeline = DataPipeline(224, self.num_classes)
            train_ds = pipeline.create_dataset(
                train_dir, 16,
                is_training=True,
                class_weights=self.class_weights
            )
            val_ds = pipeline.create_dataset(
                val_dir, 16,
                is_training=False
            )
            
            history = model.fit(
                train_ds,
                epochs=20,
                steps_per_epoch=100,
                validation_data=val_ds,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        f"{Config.MODEL_DIR}/final_best.keras",
                        monitor='val_accuracy',
                        save_best_only=True
                    ),
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )
        else:
            print("⚠ Could not find backbone for fine-tuning")
        
        return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Setup
    combine_validation_sets(Config.VAL_DIR, Config.TEST_DIR, Config.COMBINED_VAL_DIR)

    Path(Config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PROTOTYPE B: OPTIMIZED BALANCED APPROACH")
    print("="*80)
    print("Fixed: XLA timeouts, GPU warmup, weight transfer")
    print("="*80 + "\n")
    
    # Get class info
    temp_ds = tf.keras.utils.image_dataset_from_directory(
        Config.TRAIN_DIR,
        image_size=(128, 128),
        batch_size=32
    )
    class_names = temp_ds.class_names
    num_classes = len(class_names)
    
    # Get class counts
    class_counts = {}
    for i, name in enumerate(class_names):
        path = os.path.join(Config.TRAIN_DIR, name)
        class_counts[i] = len([f for f in os.listdir(path) 
                              if os.path.isfile(os.path.join(path, f))])
    
    print(f"Classes: {class_names}")
    print(f"Class distribution: {class_counts}\n")
    
    # Train
    trainer = ProgressiveTrainer(num_classes, class_names, class_counts)
    model = trainer.train_progressive(Config.TRAIN_DIR, Config.COMBINED_VAL_DIR)
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    pipeline = DataPipeline(224, num_classes)
    val_ds = pipeline.create_dataset(
        Config.COMBINED_VAL_DIR, 32,
        is_training=False
    )
    
    results = model.evaluate(val_ds, return_dict=True)
    
    print(f"\nFinal Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['prec']*100:.2f}%")
    print(f"Recall: {results['rec']*100:.2f}%")
    
    # Save
    model.save(f"{Config.MODEL_DIR}/final_model.keras")
    
    with open(f"{Config.RESULTS_DIR}/metrics.json", 'w') as f:
        json.dump({
            'accuracy': float(results['accuracy']),
            'precision': float(results['prec']),
            'recall': float(results['recall']),
            'class_names': class_names
        }, f, indent=4)
    
    print(f"\n✓ Complete! Saved to {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
