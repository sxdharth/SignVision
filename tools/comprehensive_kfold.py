import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
X_PATH = os.path.join(DATA_DIR, 'X_video_call.npy')
Y_PATH = os.path.join(DATA_DIR, 'y_video_call.npy')
CLASSES_PATH = os.path.join(DATA_DIR, 'video_call_classes.json')

def build_conv_gru(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.5),
        GRU(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        GRU(64, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def augment_data(X_train, y_train):
    X_aug, y_aug = [], []
    for i in range(len(X_train)):
        seq, label = X_train[i], y_train[i]
        X_aug.append(seq)
        y_aug.append(label)
        X_aug.append(seq + np.random.normal(0, 0.02, seq.shape))
        y_aug.append(label)
        X_aug.append(seq + np.random.normal(0, 0.04, seq.shape))
        y_aug.append(label)
        shift = np.random.randint(1, 3)
        shifted = np.roll(seq, shift, axis=0)
        shifted[:shift] = seq[0]
        X_aug.append(shifted)
        y_aug.append(label)
        dropout_seq = seq.copy()
        if np.random.random() > 0.5: dropout_seq[:, 99:162] = 0
        else: dropout_seq[:, 162:225] = 0
        X_aug.append(dropout_seq)
        y_aug.append(label)
    return np.array(X_aug), np.array(y_aug)

def main():
    X = np.load(X_PATH)
    y_raw = np.load(Y_PATH)
    with open(CLASSES_PATH, 'r') as f:
        class_map = json.load(f)
    
    # invert map to get class names array correctly matching categorical ordering
    inv_map = {v: k for k, v in class_map.items()}
    class_names = [inv_map[i] for i in range(len(inv_map))]
    num_classes = len(class_names)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_train_acc, fold_test_acc = [], []
    all_y_actual, all_y_pred = [], []
    
    print("="*60)
    print(" COMPREHENSIVE K-FOLD CROSS-VALIDATION (LEAK-FREE)")
    print("="*60)
    print(f"Total Base Samples: {X.shape[0]} | Classes: {num_classes}")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_raw), 1):
        print(f"\n--- FOLD {fold}/5 ---")
        
        # 1. Split BEFORE any data manipulation
        X_train_raw, X_test = X[train_idx], X[test_idx]
        y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]
        
        # 2. Augment ONLY the train fold
        print(f"  Augmenting Train split ({len(X_train_raw)} -> 5x)...")
        X_train, y_train_aug = augment_data(X_train_raw, y_train_raw)
        
        y_train = to_categorical(y_train_aug, num_classes)
        y_test = to_categorical(y_test_raw, num_classes)
        
        model = build_conv_gru((X.shape[1], X.shape[2]), num_classes)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        print("  Training Model... this may take a moment.")
        model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)
        
        loss_train, acc_train = model.evaluate(X_train, y_train, verbose=0)
        loss_test, acc_test = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"  > Train Accuracy: {acc_train*100:.2f}%")
        print(f"  > Test Accuracy:  {acc_test*100:.2f}%")
        
        fold_train_acc.append(acc_train * 100)
        fold_test_acc.append(acc_test * 100)
        
        preds = model.predict(X_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        
        all_y_actual.extend(y_test_raw)
        all_y_pred.extend(y_pred)
        
        tf.keras.backend.clear_session()
        
    print("\n" + "="*60)
    print(" FINAL CROSS-VALIDATION RESULTS ")
    print("="*60)
    print(f"Mean Train Accuracy: {np.mean(fold_train_acc):.2f}% (Std: ±{np.std(fold_train_acc):.2f}%)")
    print(f"Mean Test Accuracy:  {np.mean(fold_test_acc):.2f}% (Std: ±{np.std(fold_test_acc):.2f}%)")
    
    print("\n--- AGGREGATED CLASSIFICATION REPORT ---")
    report = classification_report(all_y_actual, all_y_pred, target_names=class_names)
    print(report)
    
    print("--- AGGREGATED CONFUSION MATRIX ---")
    cm = confusion_matrix(all_y_actual, all_y_pred)
    header = "          " + "".join([f"{name:>10}" for name in class_names])
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = f"{row_name:>10} " + "".join([f"{val:>10}" for val in cm[i]])
        print(row_str)

    with open('comprehensive_eval_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"COMPREHENSIVE K-FOLD RESULTS\n==========================\n")
        f.write(f"Mean Train Accuracy: {np.mean(fold_train_acc):.2f}% (Std: ±{np.std(fold_train_acc):.2f}%)\n")
        f.write(f"Mean Test Accuracy:  {np.mean(fold_test_acc):.2f}% (Std: ±{np.std(fold_test_acc):.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print("\nResults mapped and written to comprehensive_eval_results.txt")

if __name__ == '__main__':
    main()
