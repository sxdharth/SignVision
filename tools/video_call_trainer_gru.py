import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
MODEL_DIR = os.path.join(ROOT_DIR, "Models")

X_PATH = os.path.join(DATA_DIR, 'X_video_call.npy')
y_PATH = os.path.join(DATA_DIR, 'y_video_call.npy')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'video_call_model_gru.h5')

os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        
        GRU(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        GRU(64, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def main():
    print("Loading specialized Video Call data for GRU...")
    try:
        X = np.load(X_PATH)
        y = np.load(y_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {X_PATH} or {y_PATH}. Run merger script first.")
        return

    print(f"Loaded {len(X)} samples.")
    
    y_cat = to_categorical(y).astype(int)
    num_classes = y_cat.shape[1]
    print(f"Number of classes: {num_classes}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
    
    # --- DATA AUGMENTATION ---
    print("Augmenting training data to prevent overfitting... (No Leakage)")
    X_train_aug, y_train_aug = [], []
    for i in range(len(X_train)):
        seq = X_train[i]
        label = y_train[i]
        
        X_train_aug.append(seq)
        y_train_aug.append(label)
        
        X_train_aug.append(seq + np.random.normal(0, 0.02, seq.shape))
        y_train_aug.append(label)
        
        X_train_aug.append(seq + np.random.normal(0, 0.04, seq.shape))
        y_train_aug.append(label)
        
        shift = np.random.randint(1, 3)
        shifted = np.roll(seq, shift, axis=0)
        shifted[:shift] = seq[0]
        X_train_aug.append(shifted)
        y_train_aug.append(label)
        
        dropout_seq = seq.copy()
        if np.random.random() > 0.5:
            dropout_seq[:, 99:162] = 0
        else:
            dropout_seq[:, 162:225] = 0
        X_train_aug.append(dropout_seq)
        y_train_aug.append(label)
        
    X_train = np.array(X_train_aug)
    y_train = np.array(y_train_aug)
    print(f"After augmentation: {len(X_train)} training samples")
    # --------------------------

    input_shape = (X.shape[1], X.shape[2])
    print(f"Input Shape for Conv1D+GRU: {input_shape}")
    
    model = build_model(input_shape, num_classes)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_categorical_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]

    print("\nStarting Fast GRU Training...")
    history = model.fit(
        X_train, y_train,
        epochs=150, 
        batch_size=32, 
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    print(f"\nTraining Complete! Best model saved to: {MODEL_SAVE_PATH}")
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Validation Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
