"""
SignVision — Comprehensive Model Evaluation Suite
===================================================
Trains both Conv1D+GRU and Bi-LSTM from scratch with an 80/20 split,
then runs 8 diagnostic tests and saves all charts to docs/evaluation_charts/.

Tests:
  1. Classification Report (Precision / Recall / F1 per class)
  2. Confusion Matrix Heatmaps
  3. Training Curves (Loss + Accuracy)
  4. t-SNE Visualization of Learned Feature Space
  5. Confidence Distribution (correct vs incorrect predictions)
  6. Per-Class Accuracy Bar Chart
  7. Model Complexity Comparison (Params + Inference Speed)
  8. Misclassification Analysis (what gets confused with what)
"""

import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.manifold import TSNE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Bidirectional, Conv1D,
                                      MaxPooling1D, GRU, Dropout, Input)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# ──────────────────── PATHS ────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'Data')
OUT  = os.path.join(ROOT, 'docs', 'evaluation_charts')
os.makedirs(OUT, exist_ok=True)

X_PATH = os.path.join(DATA, 'X_video_call.npy')
Y_PATH = os.path.join(DATA, 'y_video_call.npy')
CLASSES_PATH = os.path.join(DATA, 'video_call_classes.json')

# ──────────────────── STYLE ────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#0e1117',
    'axes.edgecolor': '#333',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'grid.color': '#222',
    'font.size': 11,
    'font.family': 'sans-serif',
})

COLORS = {
    'gru': '#38bdf8',     # Signature blue
    'lstm': '#f472b6',    # Pink
    'correct': '#22c55e', # Green
    'wrong': '#ef4444',   # Red
}

# ──────────────────── MODEL BUILDERS ────────────────────
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
    model.compile(optimizer=Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model

def build_bilstm(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='relu'),
                      input_shape=input_shape),
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
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model

# ──────────────────── TRAIN ────────────────────
def train_model(builder, name, X_train, y_train_cat, X_test, y_test_cat, input_shape, num_classes):
    print(f"\n{'='*50}")
    print(f"  Training {name}")
    print(f"{'='*50}")
    model = builder(input_shape, num_classes)
    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test_cat),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  >> Final Test Accuracy: {acc*100:.2f}%")
    return model, history

# ──────────────────── TEST 1: Classification Report ────────────────────
def test_classification_report(model, X_test, y_test, class_names, name, out_dir):
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Print to console
    print(f"\n--- {name}: Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Save text file
    with open(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_classification_report.txt'), 'w') as f:
        f.write(classification_report(y_test, y_pred, target_names=class_names))
    
    return y_pred, report

# ──────────────────── TEST 2: Confusion Matrix ────────────────────
def test_confusion_matrix(y_test, y_pred, class_names, name, out_dir, color):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if 'GRU' in name else 'RdPu',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor='#333')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{name} — Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_confusion_matrix.png'), dpi=200)
    plt.close()
    print(f"  ✓ Saved confusion matrix for {name}")

# ──────────────────── TEST 3: Training Curves ────────────────────
def test_training_curves(histories, names, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for hist, name, clr in zip(histories, names, [COLORS['gru'], COLORS['lstm']]):
        h = hist.history
        epochs = range(1, len(h['loss'])+1)
        
        ax1.plot(epochs, h['loss'], '-', color=clr, alpha=0.4, linewidth=1)
        ax1.plot(epochs, h['val_loss'], '-', color=clr, linewidth=2.5, label=f'{name} (val)')
        
        ax2.plot(epochs, h['categorical_accuracy'], '-', color=clr, alpha=0.4, linewidth=1)
        ax2.plot(epochs, h['val_categorical_accuracy'], '-', color=clr, linewidth=2.5, label=f'{name} (val)')
    
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.2)
    
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 1.05)
    
    plt.suptitle('GRU vs LSTM — Training Dynamics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved training curves comparison")

# ──────────────────── TEST 4: t-SNE ────────────────────
def test_tsne(model, X_test, y_test, class_names, name, out_dir, color_base):
    # Extract features from the penultimate dense layer
    # Use a truncated model approach that works with Sequential
    penultimate_layer = model.layers[-2]
    truncated = tf.keras.Sequential(model.layers[:-1])
    # Warm up by passing a sample
    _ = truncated.predict(X_test[:1], verbose=0)
    features = truncated.predict(X_test, verbose=0)
    
    perplexity = min(30, len(X_test) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embedded = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab10', len(class_names))
    
    for i, cls in enumerate(class_names):
        mask = y_test == i
        ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=[cmap(i)], label=cls, alpha=0.8, s=60, edgecolors='white', linewidth=0.3)
    
    ax.set_title(f'{name} — t-SNE Feature Space', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_tsne.png'), dpi=200)
    plt.close()
    print(f"  ✓ Saved t-SNE for {name}")

# ──────────────────── TEST 5: Confidence Distribution ────────────────────
def test_confidence(model, X_test, y_test, name, out_dir, color):
    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)
    max_conf = probs.max(axis=1)
    
    correct_mask = y_pred == y_test
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(max_conf[correct_mask], bins=bins, alpha=0.7, color=COLORS['correct'], 
            label=f'Correct ({correct_mask.sum()})', edgecolor='none')
    ax.hist(max_conf[~correct_mask], bins=bins, alpha=0.7, color=COLORS['wrong'], 
            label=f'Incorrect ({(~correct_mask).sum()})', edgecolor='none')
    
    ax.axvline(x=0.5, color='#fff', linestyle='--', alpha=0.3, label='50% threshold')
    ax.set_title(f'{name} — Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Confidence (max softmax)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_confidence.png'), dpi=200)
    plt.close()
    print(f"  ✓ Saved confidence distribution for {name}")

# ──────────────────── TEST 6: Per-Class Accuracy ────────────────────
def test_per_class_accuracy(models_preds, y_test, class_names, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.35
    
    for i, (name, y_pred, color) in enumerate(models_preds):
        accs = []
        for cls_idx in range(len(class_names)):
            mask = y_test == cls_idx
            if mask.sum() > 0:
                accs.append((y_pred[mask] == cls_idx).mean() * 100)
            else:
                accs.append(0)
        bars = ax.bar(x + i*width - width/2, accs, width, label=name, color=color, 
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8, color='#ccc')
    
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy — GRU vs LSTM', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_class_accuracy.png'), dpi=200)
    plt.close()
    print("  ✓ Saved per-class accuracy comparison")

# ──────────────────── TEST 7: Model Complexity ────────────────────
def test_complexity(models, names, X_test, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    params = [m.count_params() for m in models]
    speeds = []
    for m in models:
        # Benchmark: time 100 inferences
        start = time.time()
        for _ in range(100):
            m.predict(X_test[:1], verbose=0)
        elapsed = time.time() - start
        speeds.append(elapsed / 100 * 1000)  # ms per inference
    
    colors = [COLORS['gru'], COLORS['lstm']]
    
    ax1.bar(names, [p/1000 for p in params], color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax1.set_title('Model Parameters (K)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Parameters (thousands)')
    for i, (n, p) in enumerate(zip(names, params)):
        ax1.text(i, p/1000 + max(params)/1000*0.02, f'{p:,}', ha='center', fontsize=9, color='#ccc')
    
    ax2.bar(names, speeds, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax2.set_title('Inference Speed (ms/sample)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Milliseconds')
    for i, (n, s) in enumerate(zip(names, speeds)):
        ax2.text(i, s + max(speeds)*0.02, f'{s:.1f}ms', ha='center', fontsize=9, color='#ccc')
    
    plt.suptitle('Model Complexity Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_complexity.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved model complexity comparison")

# ──────────────────── TEST 8: Misclassification Analysis ────────────────────
def test_misclassification(model, X_test, y_test, y_pred, class_names, name, out_dir):
    wrong_mask = y_pred != y_test
    if wrong_mask.sum() == 0:
        print(f"  ✓ {name}: No misclassifications — Perfect score!")
        return
    
    wrong_actual = y_test[wrong_mask]
    wrong_predicted = y_pred[wrong_mask]
    
    probs = model.predict(X_test[wrong_mask], verbose=0)
    
    print(f"\n--- {name}: Misclassification Analysis ({wrong_mask.sum()} errors) ---")
    print(f"{'Actual':<15} {'Predicted':<15} {'Confidence':>10}")
    print("-" * 42)
    for a, p, prob in zip(wrong_actual, wrong_predicted, probs):
        print(f"  {class_names[a]:<15} {class_names[p]:<15} {prob[p]*100:>8.1f}%")
    
    # Save confusion pairs
    pairs = {}
    for a, p in zip(wrong_actual, wrong_predicted):
        key = f"{class_names[a]} → {class_names[p]}"
        pairs[key] = pairs.get(key, 0) + 1
    
    if pairs:
        fig, ax = plt.subplots(figsize=(10, max(4, len(pairs) * 0.6)))
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
        labels, counts = zip(*sorted_pairs)
        
        ax.barh(range(len(labels)), counts, color=COLORS['wrong'], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Count')
        ax.set_title(f'{name} — Most Confused Pairs', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_misclassifications.png'), dpi=200)
        plt.close()
        print(f"  ✓ Saved misclassification chart for {name}")

# ══════════════════════════════════════════════════════════════
#                          MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  SignVision — Comprehensive Model Evaluation     ║")
    print("╚══════════════════════════════════════════════════╝")
    
    # Load data
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    with open(CLASSES_PATH) as f:
        class_map = json.load(f)
    
    class_names = [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]
    num_classes = len(class_names)
    input_shape = (X.shape[1], X.shape[2])
    
    print(f"\n  Dataset: {X.shape[0]} samples, {num_classes} classes")
    print(f"  Shape:   {X.shape}")
    print(f"  Classes: {', '.join(class_names)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # ─── Train Both Models ───
    gru_model, gru_hist = train_model(
        build_conv_gru, "Conv1D+GRU", X_train, y_train_cat, X_test, y_test_cat, input_shape, num_classes
    )
    lstm_model, lstm_hist = train_model(
        build_bilstm, "Bi-LSTM", X_train, y_train_cat, X_test, y_test_cat, input_shape, num_classes
    )
    
    # ─── Run All Tests ───
    print(f"\n{'═'*50}")
    print(f"  Running 8 Diagnostic Tests...")
    print(f"{'═'*50}")
    
    # Test 1: Classification Report
    gru_pred, gru_report = test_classification_report(gru_model, X_test, y_test, class_names, "Conv1D+GRU", OUT)
    lstm_pred, lstm_report = test_classification_report(lstm_model, X_test, y_test, class_names, "Bi-LSTM", OUT)
    
    # Test 2: Confusion Matrices
    test_confusion_matrix(y_test, gru_pred, class_names, "Conv1D+GRU", OUT, COLORS['gru'])
    test_confusion_matrix(y_test, lstm_pred, class_names, "Bi-LSTM", OUT, COLORS['lstm'])
    
    # Test 3: Training Curves
    test_training_curves([gru_hist, lstm_hist], ['Conv1D+GRU', 'Bi-LSTM'], OUT)
    
    # Test 4: t-SNE
    test_tsne(gru_model, X_test, y_test, class_names, "Conv1D+GRU", OUT, COLORS['gru'])
    test_tsne(lstm_model, X_test, y_test, class_names, "Bi-LSTM", OUT, COLORS['lstm'])
    
    # Test 5: Confidence Distribution
    test_confidence(gru_model, X_test, y_test, "Conv1D+GRU", OUT, COLORS['gru'])
    test_confidence(lstm_model, X_test, y_test, "Bi-LSTM", OUT, COLORS['lstm'])
    
    # Test 6: Per-Class Accuracy
    test_per_class_accuracy(
        [("Conv1D+GRU", gru_pred, COLORS['gru']), ("Bi-LSTM", lstm_pred, COLORS['lstm'])],
        y_test, class_names, OUT
    )
    
    # Test 7: Model Complexity
    test_complexity([gru_model, lstm_model], ['Conv1D+GRU', 'Bi-LSTM'], X_test, OUT)
    
    # Test 8: Misclassification Analysis
    test_misclassification(gru_model, X_test, y_test, gru_pred, class_names, "Conv1D+GRU", OUT)
    test_misclassification(lstm_model, X_test, y_test, lstm_pred, class_names, "Bi-LSTM", OUT)
    
    # ─── Final Summary ───
    print(f"\n\n{'#'*50}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'#'*50}")
    print(f"\n  Conv1D+GRU Test Accuracy: {gru_report['accuracy']*100:.2f}%")
    print(f"  Bi-LSTM    Test Accuracy: {lstm_report['accuracy']*100:.2f}%")
    print(f"\n  All charts saved to: {OUT}")
    print(f"  Total charts generated: 10")

if __name__ == '__main__':
    main()
