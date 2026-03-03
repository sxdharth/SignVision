import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
MODEL_DIR = os.path.join(ROOT_DIR, "Models")

X_PATH = os.path.join(DATA_DIR, 'X_smart_home.npy')
y_PATH = os.path.join(DATA_DIR, 'y_smart_home.npy')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'smart_home_model.h5')

# Ensure Model dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(input_shape, num_classes):
    """
    To combat low accuracy/high loss on small datasets, 
    we use a simpler architecture with high dropout and L2 regularization.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dropout(0.5), # Heavy dropout to prevent memorizing the 20 samples
        LSTM(32, return_sequences=False, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Use a lower learning rate for stable convergence
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def main():
    print("Loading specialized Smart Home data...")
    try:
        X = np.load(X_PATH)
        y = np.load(y_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {X_PATH} or {y_PATH}. Run merger script first.")
        return

    num_classes = len(np.unique(y))
    print(f"Loaded {len(X)} samples across {num_classes} classes.")
    
    # One-hot encode labels
    y_cat = to_categorical(y).astype(int)

    # Split: 80% train, 20% validation
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    input_shape = (X.shape[1], X.shape[2])
    print(f"Input Shape: {input_shape}")
    
    model = build_model(input_shape, num_classes)
    model.summary()

    # Callbacks to prevent overfitting and save best model
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_categorical_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
    ]

    print("\nStarting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=150, # Many epochs because EarlyStopping will catch it
        batch_size=8, # Small batch size ideal for small datasets
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    print(f"\nTraining Complete! Best model saved to: {MODEL_SAVE_PATH}")
    
    # Final eval
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Validation Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
