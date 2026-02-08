

import os
# Suppress logs
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATASET_ROOT = "/kaggle/input/dataset/dataset"
    TRAIN_DIR = f"{DATASET_ROOT}/train"
    VAL_DIR = f"{DATASET_ROOT}/val"
    TEST_DIR = f"{DATASET_ROOT}/test"
    COMBINED_VAL_DIR = "/kaggle/working/combined_validation_server"
    
    OUTPUT_DIR = "/kaggle/working/prototype_g_server"
    MODEL_DIR = f"{OUTPUT_DIR}/models"
    
    # SERVER SETTINGS (High Quality)
    INPUT_SHAPE = (224, 224, 3) 
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    RANDOM_SEED = 42

# ============================================================================
# DATA PIPELINE 
# ============================================================================

def create_dataset(directory, batch_size, is_training=True, shuffle=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory, 
        image_size=Config.INPUT_SHAPE[:2],
        batch_size=batch_size,
        label_mode='categorical', 
        shuffle=shuffle, 
        seed=Config.RANDOM_SEED if shuffle else None
    )

    # Geometric Augmentation Only (Safe for Defects)
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal_and_vertical"),
    ])

    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        if is_training:
            images = data_augmentation(images, training=True)
        # EfficientNet expects [0-255] range (it handles norm internally)
        images = tf.clip_by_value(images, 0.0, 255.0)
        return images, labels
    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training and shuffle: 
        ds = ds.shuffle(1000).repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ============================================================================
  # MODEL PART 1
# ============================================================================

def build_cnn(num_classes):
    inputs = keras.Input(shape=Config.INPUT_SHAPE)
    
    # Load EfficientNetB0 (Frozen)
    base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    base.trainable = False 
    
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x) 
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="EfficientNet_Head")
    return model

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    Path(Config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    print(f"PROTOTYPE G: SERVER-SIDE ENSEMBLE BUILD")
    print("="*60)
    
    # 1. Combine Data (Val + Test)
    if os.path.exists(Config.COMBINED_VAL_DIR): shutil.rmtree(Config.COMBINED_VAL_DIR)
    os.makedirs(Config.COMBINED_VAL_DIR)
    
    print("Combining Val/Test for robust scoring...")
    for d in [Config.VAL_DIR, Config.TEST_DIR]:
        if not os.path.exists(d): continue
        for class_name in os.listdir(d):
            src = os.path.join(d, class_name)
            dst = os.path.join(Config.COMBINED_VAL_DIR, class_name)
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f"copy_{f}"))

    # Load Data
    train_ds = create_dataset(Config.TRAIN_DIR, Config.BATCH_SIZE, is_training=True)
    val_ds = create_dataset(Config.COMBINED_VAL_DIR, Config.BATCH_SIZE, is_training=False, shuffle=False)
    
    # Get Metadata (Need class names for reports)
    temp_ds = tf.keras.utils.image_dataset_from_directory(Config.COMBINED_VAL_DIR, shuffle=False)
    class_names = temp_ds.class_names
    num_classes = len(class_names)
    
    steps = sum([len(files) for r, d, files in os.walk(Config.TRAIN_DIR)]) // Config.BATCH_SIZE
    
    # 2. TRAIN CNN (EfficientNet)
    print("\n[PART 1] Training EfficientNet...")
    cnn_model = build_cnn(num_classes)
    
    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    cnn_model.fit(
        train_ds,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps,
        validation_data=val_ds,
        callbacks=[
            keras.callbacks.ModelCheckpoint(f"{Config.MODEL_DIR}/cnn_best.keras", save_best_only=True, monitor='val_accuracy'),
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # 3. TRAIN SVM (The "Expert" Head)
    print("\n[PART 2] Training SVM Head...")
    
    # Create Feature Extractor (Remove Softmax Layer)
    feature_extractor = keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
    
    # Extract Features
    print("Extracting features from Train Set...")
    # We need a non-shuffled train set for SVM matching
    train_ds_svm = create_dataset(Config.TRAIN_DIR, Config.BATCH_SIZE, is_training=False, shuffle=False)
    
    X_train = feature_extractor.predict(train_ds_svm, verbose=1)
    y_train = np.concatenate([y for x, y in train_ds_svm], axis=0)
    y_train = np.argmax(y_train, axis=1)
    
    print("Extracting features from Validation Set...")
    X_val = feature_extractor.predict(val_ds, verbose=1)
    y_val = np.concatenate([y for x, y in val_ds], axis=0)
    y_val = np.argmax(y_val, axis=1)
    
    # Fit SVM
    print("Fitting SVM (RBF Kernel)...")
    svm_clf = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', C=10.0, probability=True, class_weight='balanced'))
    svm_clf.fit(X_train, y_train)
    
    acc_svm = svm_clf.score(X_val, y_val)
    print(f"SVM Standalone Accuracy: {acc_svm*100:.2f}%")
    
    # Save SVM
    joblib.dump(svm_clf, f"{Config.MODEL_DIR}/svm_head.pkl")
    
    # 4. ENSEMBLE VOTING (The "Disqualified" Logic)
    print("\n[PART 3] Calculating Ensemble Results...")
    
    # Get CNN Probabilities
    cnn_probs = cnn_model.predict(val_ds, verbose=0)
    
    # Get SVM Probabilities
    svm_probs = svm_clf.predict_proba(X_val)
    
    # Weighted Vote (60% SVM + 40% CNN)
    ensemble_probs = (0.4 * cnn_probs) + (0.6 * svm_probs)
    y_pred = np.argmax(ensemble_probs, axis=1)
    
    # 5. FINAL REPORT
    acc = metrics.accuracy_score(y_val, y_pred)
    print("\n" + "="*40)
    print(f"FINAL ENSEMBLE ACCURACY: {acc*100:.2f}%")
    print("="*40)
    
    print("\nClassification Report:")
    print(metrics.classification_report(y_val, y_pred, target_names=class_names))
    
    # Plot Confusion Matrix
    cm = metrics.confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Ensemble Confusion Matrix (Acc: {acc*100:.1f}%)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{Config.OUTPUT_DIR}/ensemble_matrix.png")
    plt.show()
    
    print(f"Saved: {Config.OUTPUT_DIR}/ensemble_matrix.png")

if __name__ == "__main__":
    main()
