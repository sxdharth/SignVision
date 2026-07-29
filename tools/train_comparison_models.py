import os
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,  Bidirectional, Conv1D, MaxPooling1D, GRU, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
MODEL_DIR = os.path.join(ROOT_DIR, "Models")

X_PATH = os.path.join(DATA_DIR, "X_wlasl100.npy")
Y_PATH = os.path.join(DATA_DIR, "y_wlasl100.npy")

def build_bilstm_model(input_shape, num_classes):
    """The original model architecture used in SignVision."""
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
    model.add(Dropout(0.2))
    
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def build_conv1d_gru_model(input_shape, num_classes):
    """The proposed new architecture for comparison."""
    model = Sequential()
    # Feature extraction over time using Conv1D
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    # Recurrent sequence learning via GRU
    model.add(GRU(128, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(GRU(64, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.3))
    
    # Dense classification layers
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def train_and_evaluate(model, name, X_train, X_test, y_train, y_test):
    print(f"\n{'='*50}")
    print(f"Training Model: {name}")
    print(f"{'='*50}")
    
    save_path = os.path.join(MODEL_DIR, f"{name}.h5")
    
    callbacks = [
        ModelCheckpoint(save_path, monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-5)
    ]
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2
    )
    training_time = time.time() - start_time
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    return {
        'name': name,
        'accuracy': accuracy,
        'loss': loss,
        'time': training_time,
        'history': history.history
    }

def main():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("Data files missing! Run extract script first.")
        return
        
    print("Loading datasets...")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    print(f"Data shape: X = {X.shape}, y = {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))
    
    input_shape = (X.shape[1], X.shape[2])
    num_classes = y.shape[1]
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Build models
    bilstm = build_bilstm_model(input_shape, num_classes)
    conv_gru = build_conv1d_gru_model(input_shape, num_classes)
    
    # Train
    bilstm_stats = train_and_evaluate(bilstm, "wlasl100_bilstm", X_train, X_test, y_train, y_test)
    conv_gru_stats = train_and_evaluate(conv_gru, "wlasl100_conv_gru", X_train, X_test, y_train, y_test)
    
    # Report
    print("\n\n" + "#"*40)
    print("FINAL COMPARISON REPORT")
    print("#"*40)
    
    print(f"\nModel 1: Bi-LSTM (Existing Architecture)")
    print(f"Test Accuracy : {bilstm_stats['accuracy']*100:.2f}%")
    print(f"Test Loss     : {bilstm_stats['loss']:.4f}")
    print(f"Training Time : {bilstm_stats['time']:.2f} seconds")
    
    print(f"\nModel 2: Conv1D + GRU (New Architecture)")
    print(f"Test Accuracy : {conv_gru_stats['accuracy']*100:.2f}%")
    print(f"Test Loss     : {conv_gru_stats['loss']:.4f}")
    print(f"Training Time : {conv_gru_stats['time']:.2f} seconds")
    
    # Save text report
    report_path = os.path.join(ROOT_DIR, "wlasl_comparison_report.txt")
    with open(report_path, "w") as f:
        f.write("SignVision - WLASL Top 100 Comparison Report\n")
        f.write("="*50 + "\n\n")
        f.write("Model 1: Bidirectional LSTM (Current)\n")
        f.write(f"Accuracy: {bilstm_stats['accuracy']*100:.2f}%\n")
        f.write(f"Time: {bilstm_stats['time']:.2f} s\n\n")
        f.write("Model 2: Conv1D + GRU (Proposed)\n")
        f.write(f"Accuracy: {conv_gru_stats['accuracy']*100:.2f}%\n")
        f.write(f"Time: {conv_gru_stats['time']:.2f} s\n")
        
    print(f"\nComparison report saved to {report_path}")

if __name__ == "__main__":
    main()
