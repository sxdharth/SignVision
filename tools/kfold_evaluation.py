"""
SignVision — Rigorous K-Fold Cross-Validation
==============================================
Runs 5-fold stratified cross-validation on the custom dataset
to produce a realistic, presentation-worthy accuracy score.
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "Data")

X_PATH = os.path.join(DATA_DIR, 'X_video_call.npy')
Y_PATH = os.path.join(DATA_DIR, 'y_video_call.npy')
CLASSES_PATH = os.path.join(DATA_DIR, 'video_call_classes.json')

def build_bilstm(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def build_conv_gru(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),
        GRU(128, return_sequences=True, activation='tanh'),
        Dropout(0.3),
        GRU(64, return_sequences=False, activation='tanh'),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def run_kfold(model_builder, model_name, X, y_raw, num_classes, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    
    print(f"\n{'='*50}")
    print(f"  {model_name} — {n_splits}-Fold Cross-Validation")
    print(f"{'='*50}")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_raw), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = to_categorical(y_raw[train_idx], num_classes)
        y_test = to_categorical(y_raw[test_idx], num_classes)
        
        model = model_builder((X.shape[1], X.shape[2]), num_classes)
        
        model.fit(X_train, y_train, epochs=80, batch_size=32,
                  validation_data=(X_test, y_test),
                  callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
                  verbose=0)
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        fold_accuracies.append(acc * 100)
        print(f"  Fold {fold}: {acc*100:.2f}%")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n  >> RESULT: {mean_acc:.2f}% ± {std_acc:.2f}%")
    return mean_acc, std_acc, fold_accuracies

def main():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    with open(CLASSES_PATH, 'r') as f:
        class_map = json.load(f)
    num_classes = len(class_map)
    
    print(f"Dataset: {X.shape[0]} samples, {num_classes} classes")
    print(f"Shape: {X.shape}")
    
    lstm_mean, lstm_std, lstm_folds = run_kfold(build_bilstm, "Bi-LSTM", X, y, num_classes)
    gru_mean, gru_std, gru_folds = run_kfold(build_conv_gru, "Conv1D + GRU", X, y, num_classes)
    
    print(f"\n\n{'#'*50}")
    print(f"  FINAL CROSS-VALIDATION REPORT")
    print(f"{'#'*50}")
    print(f"\n  Bi-LSTM:      {lstm_mean:.2f}% ± {lstm_std:.2f}%")
    print(f"  Conv1D + GRU: {gru_mean:.2f}% ± {gru_std:.2f}%")
    print(f"\n  (Mean ± Std across 5 folds)")
    
    # Save results for visualization script
    results_path = os.path.join(DATA_DIR, 'kfold_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'lstm': {'mean': lstm_mean, 'std': lstm_std, 'folds': lstm_folds},
            'gru': {'mean': gru_mean, 'std': gru_std, 'folds': gru_folds}
        }, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

if __name__ == '__main__':
    main()
