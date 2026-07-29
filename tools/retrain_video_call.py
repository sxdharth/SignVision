"""
retrain_video_call.py
=====================
One-click script: merges raw webcam recordings → retrains the model.
Run AFTER `video_call_data_recorder.py` has collected all samples.

Usage (from project root):
    d:\SignVision_S8_V2\.venv\Scripts\python.exe tools/retrain_video_call.py
"""

import os
import json
import numpy as np
import sys
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Primary: user-recorded data in Video_Call_Raw
# Fallback: pre-processed data in Custom_Processed
RAW_DIR_PRIMARY = os.path.join(ROOT_DIR, 'Data', 'Video_Call_Raw')
RAW_DIR_FALLBACK = os.path.join(ROOT_DIR, 'Data', 'Custom_Processed')
RAW_DIR  = RAW_DIR_PRIMARY if os.path.exists(RAW_DIR_PRIMARY) and len([d for d in os.listdir(RAW_DIR_PRIMARY) if os.path.isdir(os.path.join(RAW_DIR_PRIMARY, d)) and len(os.listdir(os.path.join(RAW_DIR_PRIMARY, d))) > 0]) > 0 else RAW_DIR_FALLBACK
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
MODEL_DIR = os.path.join(ROOT_DIR, 'Models')

X_OUT       = os.path.join(DATA_DIR, 'X_video_call.npy')
y_OUT       = os.path.join(DATA_DIR, 'y_video_call.npy')
CLASSES_OUT = os.path.join(DATA_DIR, 'video_call_classes.json')
MODEL_OUT   = os.path.join(MODEL_DIR, 'video_call_model.h5')

# ── 1. MERGE RAW RECORDINGS ─────────────────────────────────────────────────

def merge():
    print("\n[1/2] Merging raw recordings from Data/Video_Call_Raw/ ...\n")

    if not os.path.exists(RAW_DIR):
        print(f"ERROR: Raw data directory not found: {RAW_DIR}")
        print("Please run video_call_data_recorder.py first.")
        sys.exit(1)

    class_names = sorted([d for d in os.listdir(RAW_DIR)
                          if os.path.isdir(os.path.join(RAW_DIR, d))])
    if not class_names:
        print("ERROR: No class folders found in Video_Call_Raw/")
        sys.exit(1)

    class_map = {name: idx for idx, name in enumerate(class_names)}
    print(f"Found {len(class_names)} classes: {class_names}")

    X_all, y_all = [], []
    for name, idx in class_map.items():
        class_dir = os.path.join(RAW_DIR, name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        print(f"  {name}: {len(files)} samples")
        for f in files:
            seq = np.load(os.path.join(class_dir, f))
            if seq.shape == (30, 225):
                X_all.append(seq)
                y_all.append(idx)
            else:
                print(f"    WARNING: Skipping {f} (bad shape {seq.shape})")

    X = np.array(X_all)
    y = np.array(y_all)
    print(f"\nTotal samples: {len(X)}  |  Shape: {X.shape}")

    np.save(X_OUT, X)
    np.save(y_OUT, y)
    with open(CLASSES_OUT, 'w') as f:
        json.dump(class_map, f, indent=4)

    print(f"Saved X → {X_OUT}")
    print(f"Saved y → {y_OUT}")
    print(f"Saved classes → {CLASSES_OUT}")
    return X, y, len(class_names)


# ── 2. RETRAIN MODEL ─────────────────────────────────────────────────────────

def retrain(X, y, num_classes):
    print("\n[2/2] Retraining model ...\n")
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.regularizers import l2
    from sklearn.model_selection import train_test_split

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Backup the old model just in case
    if os.path.exists(MODEL_OUT):
        backup = MODEL_OUT.replace('.h5', '_backup.h5')
        shutil.copy(MODEL_OUT, backup)
        print(f"Old model backed up → {backup}")

    y_cat = to_categorical(y, num_classes=num_classes).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y)

    print(f"Train (raw): {len(X_train)}  |  Test: {len(X_test)}")

    # ── DATA AUGMENTATION ────────────────────────────────────────────────
    # The model was overfitting and failing to generalize to the live pipeline.
    # Augmentation makes the model robust (Applied strictly to train set to avoid leakage).
    print("Augmenting strictly the training data...")
    X_train_aug, y_train_aug = [], []
    for i in range(len(X_train)):
        seq = X_train[i]
        label = y_train[i]

        # Original sample
        X_train_aug.append(seq)
        y_train_aug.append(label)

        # Augmentation 1: Gaussian noise (simulates JPEG + MediaPipe variance)
        noisy = seq + np.random.normal(0, 0.02, seq.shape)
        X_train_aug.append(noisy)
        y_train_aug.append(label)

        # Augmentation 2: Stronger noise
        noisy2 = seq + np.random.normal(0, 0.04, seq.shape)
        X_train_aug.append(noisy2)
        y_train_aug.append(label)

        # Augmentation 3: Temporal jitter (shift frames by 1-2 positions)
        shift = np.random.randint(1, 3)
        shifted = np.roll(seq, shift, axis=0)
        shifted[:shift] = seq[0]  # Fill wrapped frames with first frame
        X_train_aug.append(shifted)
        y_train_aug.append(label)

        # Augmentation 4: Random landmark dropout (zero out one hand randomly)
        dropout_seq = seq.copy()
        if np.random.random() > 0.5:
            dropout_seq[:, 99:162] = 0   # Zero left hand
        else:
            dropout_seq[:, 162:225] = 0  # Zero right hand
        X_train_aug.append(dropout_seq)
        y_train_aug.append(label)

    X_train = np.array(X_train_aug)
    y_train = np.array(y_train_aug)
    print(f"After augmentation: {len(X_train)} training samples (5x original)")
    # ─────────────────────────────────────────────────────────────────────

    # Increased regularization to fight overfitting:
    # - Dropout 0.4 → 0.5
    # - L2 0.005 → 0.01
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu',
             input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        LSTM(64, return_sequences=False, activation='relu',
             kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_OUT, monitor='val_categorical_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, min_lr=1e-5, verbose=1)
    ]

    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Validation Accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")
    print(f"Model saved → {MODEL_OUT}")


if __name__ == '__main__':
    X, y, num_classes = merge()
    retrain(X, y, num_classes)
    print("\nDone! Restart the server to load the new model.")

