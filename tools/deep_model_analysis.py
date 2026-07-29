"""
SignVision — Deep Model Interpretability Suite
================================================
Goes FAR beyond accuracy metrics. This script dissects both GRU and LSTM
to answer: "HOW does each model actually make its decisions?"

Deep Tests:
  1. Gradient Saliency Maps — Which landmarks have the strongest influence on predictions
  2. Temporal Frame Importance — Which frames in the 30-frame window matter most
  3. Body-Part Attribution — Hands vs Face vs Pose contribution breakdown
  4. Landmark Occlusion Sensitivity — Systematically mask each body region
  5. Layer-wise Feature Evolution — PCA of activations through each layer
  6. Per-Class Prototype Analysis — What the "ideal" sample looks like to each model
  7. Noise Robustness Testing — How much noise before predictions flip
  8. Decision Boundary Probing — Interpolate between classes to find the boundary
"""

import os, json, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Bidirectional, Conv1D,
                                      MaxPooling1D, GRU, Dropout)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# ──────────────────── PATHS ────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'Data')
OUT  = os.path.join(ROOT, 'docs', 'deep_analysis')
os.makedirs(OUT, exist_ok=True)

X_PATH = os.path.join(DATA, 'X_video_call.npy')
Y_PATH = os.path.join(DATA, 'y_video_call.npy')
CLASSES_PATH = os.path.join(DATA, 'video_call_classes.json')

# ──────────────────── STYLE ────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0e1117', 'axes.facecolor': '#0e1117',
    'axes.edgecolor': '#333', 'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0', 'xtick.color': '#aaa', 'ytick.color': '#aaa',
    'grid.color': '#222', 'font.size': 11, 'font.family': 'sans-serif',
})
C_GRU, C_LSTM = '#38bdf8', '#f472b6'

# MediaPipe landmark groups (225 features = 33 pose*4 + 21 left hand*3 + 21 right hand*3 + 468 face → but ours is 225)
# 225 = 33*3 (pose xyz) = 99 + 21*3 (left hand) = 63 + 21*3 (right hand) = 63   → 99+63+63 = 225
BODY_PARTS = {
    'Pose (Body)':     (0, 99),     # 33 pose landmarks * 3
    'Left Hand':       (99, 162),   # 21 hand landmarks * 3  
    'Right Hand':      (162, 225),  # 21 hand landmarks * 3
}

# ──────────────────── MODEL BUILDERS ────────────────────
def build_conv_gru(shape, n):
    m = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=shape),
        Conv1D(128, 3, activation='relu'), MaxPooling1D(2), Dropout(0.3),
        GRU(128, return_sequences=True, activation='tanh'), Dropout(0.3),
        GRU(64, return_sequences=False, activation='tanh'), Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(n, activation='softmax')
    ])
    m.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return m

def build_bilstm(shape, n):
    m = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=shape),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=True, activation='relu')), Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'), Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(n, activation='softmax')
    ])
    m.compile(optimizer=Adam(0.001, clipnorm=1.0), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return m

def train(builder, name, Xtr, ytr, Xte, yte, shape, n):
    print(f"\n{'='*50}\n  Training {name}\n{'='*50}")
    m = builder(shape, n)
    h = m.fit(Xtr, ytr, epochs=100, batch_size=32, validation_data=(Xte, yte),
              callbacks=[EarlyStopping('val_loss', patience=10, restore_best_weights=True)], verbose=1)
    loss, acc = m.evaluate(Xte, yte, verbose=0)
    print(f"  >> {name} Accuracy: {acc*100:.2f}%")
    return m, h

# ═══════════════════════════════════════════════════════════
#  TEST 1: GRADIENT SALIENCY — Which landmarks drive predictions?
# ═══════════════════════════════════════════════════════════
def test_gradient_saliency(model, X_test, y_test, class_names, name, out_dir):
    """Compute input gradients for each class to find which landmarks
    the model 'looks at' when making predictions."""
    print(f"\n  [TEST 1] Gradient Saliency for {name}...")
    
    num_classes = len(class_names)
    # Accumulate gradient magnitude per class, averaged over test samples
    class_saliency = np.zeros((num_classes, X_test.shape[2]))  # (classes, 225)
    class_counts = np.zeros(num_classes)
    
    for cls in range(num_classes):
        mask = y_test == cls
        samples = X_test[mask]
        if len(samples) == 0:
            continue
        
        x_tensor = tf.constant(samples, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            preds = model(x_tensor, training=False)
            target_score = preds[:, cls]
        
        grads = tape.gradient(target_score, x_tensor)  # (N, 30, 225)
        # Average absolute gradient across all frames and samples
        sal = np.abs(grads.numpy()).mean(axis=(0, 1))  # (225,)
        class_saliency[cls] = sal
        class_counts[cls] = len(samples)
    
    # Normalize each class saliency to [0, 1]
    for c in range(num_classes):
        mx = class_saliency[c].max()
        if mx > 0:
            class_saliency[c] /= mx
    
    # --- Chart 1A: Heatmap of landmark importance per class ---
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(class_saliency, aspect='auto', cmap='magma', interpolation='bilinear')
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Landmark Feature Index (0-224)')
    ax.set_title(f'{name} — Gradient Saliency per Class', fontsize=14, fontweight='bold')
    
    # Mark body part boundaries
    for part_name, (start, end) in BODY_PARTS.items():
        ax.axvline(x=start, color='white', linewidth=0.5, alpha=0.5)
        ax.text(start + (end-start)/2, -0.7, part_name, ha='center', fontsize=8, color='#aaa')
    
    plt.colorbar(im, ax=ax, label='Normalized Gradient Magnitude')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_gradient_saliency.png'), dpi=200)
    plt.close()
    
    # --- Chart 1B: Body-part importance per class (bar chart) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(num_classes)
    width = 0.25
    
    for i, (part_name, (start, end)) in enumerate(BODY_PARTS.items()):
        means = [class_saliency[c, start:end].mean() for c in range(num_classes)]
        ax.bar(x + i*width - width, means, width, label=part_name, alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylabel('Mean Gradient Importance')
    ax.set_title(f'{name} — Body Part Attribution per Sign', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_body_part_attribution.png'), dpi=200)
    plt.close()
    print(f"  ✓ Gradient saliency + body-part attribution saved for {name}")

# ═══════════════════════════════════════════════════════════
#  TEST 2: TEMPORAL FRAME IMPORTANCE — When does the model pay attention?
# ═══════════════════════════════════════════════════════════
def test_temporal_importance(model, X_test, y_test, class_names, name, out_dir):
    """Which of the 30 frames in the sequence are most important?"""
    print(f"  [TEST 2] Temporal Frame Importance for {name}...")
    
    num_classes = len(class_names)
    # (classes, 30_frames)
    frame_importance = np.zeros((num_classes, X_test.shape[1]))
    
    for cls in range(num_classes):
        mask = y_test == cls
        samples = X_test[mask]
        if len(samples) == 0:
            continue
        
        x_tensor = tf.constant(samples, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            preds = model(x_tensor, training=False)
            target_score = preds[:, cls]
        
        grads = tape.gradient(target_score, x_tensor)
        # Sum absolute gradient across all landmarks, average across samples
        fi = np.abs(grads.numpy()).sum(axis=2).mean(axis=0)  # (30,)
        frame_importance[cls] = fi / fi.max() if fi.max() > 0 else fi
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(frame_importance, aspect='auto', cmap='inferno', interpolation='bilinear')
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Frame Index (0 = Start, 29 = End)')
    ax.set_title(f'{name} — Temporal Attention Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized Frame Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_temporal_importance.png'), dpi=200)
    plt.close()
    print(f"  ✓ Temporal importance saved for {name}")

# ═══════════════════════════════════════════════════════════
#  TEST 3: LANDMARK OCCLUSION SENSITIVITY
# ═══════════════════════════════════════════════════════════
def test_occlusion(model, X_test, y_test_cat, class_names, name, out_dir):
    """Mask each body part and measure accuracy drop."""
    print(f"  [TEST 3] Occlusion Sensitivity for {name}...")
    
    baseline_loss, baseline_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    
    results = {}
    for part_name, (start, end) in BODY_PARTS.items():
        X_occluded = X_test.copy()
        X_occluded[:, :, start:end] = 0  # Zero out this body part
        loss, acc = model.evaluate(X_occluded, y_test_cat, verbose=0)
        drop = (baseline_acc - acc) * 100
        results[part_name] = {'acc': acc * 100, 'drop': drop}
        print(f"    {part_name}: {acc*100:.1f}% (dropped {drop:.1f}%)")
    
    # Also test: mask ALL landmarks except each part (isolation test)
    isolation = {}
    for part_name, (start, end) in BODY_PARTS.items():
        X_isolated = np.zeros_like(X_test)
        X_isolated[:, :, start:end] = X_test[:, :, start:end]
        loss, acc = model.evaluate(X_isolated, y_test_cat, verbose=0)
        isolation[part_name] = acc * 100
        print(f"    {part_name} ONLY: {acc*100:.1f}%")
    
    # Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    parts = list(BODY_PARTS.keys())
    colors = ['#f59e0b', '#10b981', '#8b5cf6']
    
    drops = [results[p]['drop'] for p in parts]
    ax1.bar(parts, drops, color=colors, alpha=0.85, edgecolor='white')
    ax1.set_title(f'{name} — Accuracy Drop When Masked', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy Drop (%)')
    for i, d in enumerate(drops):
        ax1.text(i, d + 0.5, f'{d:.1f}%', ha='center', fontsize=10, color='#ccc')
    ax1.grid(True, axis='y', alpha=0.2)
    
    iso_vals = [isolation[p] for p in parts]
    ax2.bar(parts, iso_vals, color=colors, alpha=0.85, edgecolor='white')
    ax2.set_title(f'{name} — Accuracy Using ONLY This Part', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.axhline(y=baseline_acc*100, color='white', linestyle='--', alpha=0.3, label=f'Baseline: {baseline_acc*100:.1f}%')
    for i, v in enumerate(iso_vals):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10, color='#ccc')
    ax2.legend(); ax2.grid(True, axis='y', alpha=0.2)
    
    plt.suptitle(f'Occlusion Sensitivity Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_occlusion.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Occlusion sensitivity saved for {name}")

# ═══════════════════════════════════════════════════════════
#  TEST 4: LAYER-WISE FEATURE EVOLUTION (PCA through layers)
# ═══════════════════════════════════════════════════════════
def test_layer_evolution(model, X_test, y_test, class_names, name, out_dir):
    """PCA of activations at each layer to see how the model transforms data."""
    print(f"  [TEST 4] Layer-wise Feature Evolution for {name}...")
    
    # Get outputs of key layers (skip Dropout layers)
    key_layers = [l for l in model.layers if not isinstance(l, Dropout)]
    n_layers = min(len(key_layers), 6)  # Show up to 6 layers
    selected = [key_layers[int(i * len(key_layers) / n_layers)] for i in range(n_layers)]
    
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4))
    if n_layers == 1: axes = [axes]
    
    cmap = plt.cm.get_cmap('tab10', len(class_names))
    
    for idx, layer in enumerate(selected):
        try:
            truncated = tf.keras.Sequential(model.layers[:model.layers.index(layer)+1])
            _ = truncated.predict(X_test[:1], verbose=0)
            feats = truncated.predict(X_test, verbose=0)
        except Exception:
            continue
        
        # Flatten if needed
        if len(feats.shape) > 2:
            feats = feats.reshape(feats.shape[0], -1)
        
        # PCA to 2D
        if feats.shape[1] >= 2:
            pca = PCA(n_components=2)
            embedded = pca.fit_transform(feats)
            
            ax = axes[idx]
            for c, cls in enumerate(class_names):
                mask = y_test == c
                ax.scatter(embedded[mask, 0], embedded[mask, 1], c=[cmap(c)], 
                          s=15, alpha=0.7, label=cls if idx == 0 else None)
            
            layer_type = layer.__class__.__name__
            ax.set_title(f'L{idx+1}: {layer_type}', fontsize=10, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
    
    if n_layers > 0:
        axes[0].legend(fontsize=6, loc='best', framealpha=0.3)
    plt.suptitle(f'{name} — Feature Separation Through Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_layer_evolution.png'), dpi=200)
    plt.close()
    print(f"  ✓ Layer evolution saved for {name}")

# ═══════════════════════════════════════════════════════════
#  TEST 5: NOISE ROBUSTNESS CURVE
# ═══════════════════════════════════════════════════════════
def test_noise_robustness(models, names, X_test, y_test_cat, out_dir):
    """Add increasing Gaussian noise and track accuracy decay."""
    print(f"  [TEST 5] Noise Robustness Curve...")
    
    noise_levels = np.arange(0, 2.1, 0.1)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model, name, color in zip(models, names, [C_GRU, C_LSTM]):
        accs = []
        for sigma in noise_levels:
            X_noisy = X_test + np.random.normal(0, sigma, X_test.shape)
            _, acc = model.evaluate(X_noisy, y_test_cat, verbose=0)
            accs.append(acc * 100)
        
        ax.plot(noise_levels, accs, '-o', color=color, linewidth=2.5, markersize=4, label=name)
        # Find the 50% accuracy threshold
        for i, a in enumerate(accs):
            if a < 50:
                ax.axvline(x=noise_levels[i], color=color, linestyle='--', alpha=0.3)
                ax.text(noise_levels[i], 52, f'σ={noise_levels[i]:.1f}', color=color, fontsize=9)
                break
    
    ax.axhline(y=100/len(set(y_test_cat.argmax(axis=1))), color='#666', linestyle=':', label='Random Chance')
    ax.set_xlabel('Noise Standard Deviation (σ)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Noise Robustness — How Much Noise Before Predictions Break?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'noise_robustness.png'), dpi=200)
    plt.close()
    print(f"  ✓ Noise robustness curve saved")

# ═══════════════════════════════════════════════════════════
#  TEST 6: FRAME DROPOUT ROBUSTNESS
# ═══════════════════════════════════════════════════════════
def test_frame_dropout(models, names, X_test, y_test_cat, out_dir):
    """Remove increasing numbers of frames and track accuracy."""
    print(f"  [TEST 6] Frame Dropout Robustness...")
    
    total_frames = X_test.shape[1]  # 30
    drop_counts = list(range(0, total_frames, 2))  # 0, 2, 4, ... 28
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model, name, color in zip(models, names, [C_GRU, C_LSTM]):
        accs = []
        for n_drop in drop_counts:
            X_dropped = X_test.copy()
            if n_drop > 0:
                drop_indices = np.random.choice(total_frames, n_drop, replace=False)
                X_dropped[:, drop_indices, :] = 0
            _, acc = model.evaluate(X_dropped, y_test_cat, verbose=0)
            accs.append(acc * 100)
        
        ax.plot(drop_counts, accs, '-o', color=color, linewidth=2.5, markersize=5, label=name)
    
    ax.set_xlabel('Number of Frames Zeroed Out (of 30)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Frame Dropout — How Many Frames Can We Lose?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'frame_dropout.png'), dpi=200)
    plt.close()
    print(f"  ✓ Frame dropout curve saved")

# ═══════════════════════════════════════════════════════════
#  TEST 7: PER-CLASS SOFTMAX PROFILE 
# ═══════════════════════════════════════════════════════════
def test_softmax_profiles(model, X_test, y_test, class_names, name, out_dir):
    """For each class, show the full softmax distribution — reveals 
    which OTHER classes the model considers as alternatives."""
    print(f"  [TEST 7] Softmax Profiles for {name}...")
    
    num_classes = len(class_names)
    probs = model.predict(X_test, verbose=0)
    
    fig, axes = plt.subplots(1, num_classes, figsize=(3.5*num_classes, 5))
    
    for cls in range(num_classes):
        mask = y_test == cls
        cls_probs = probs[mask]  # (N, num_classes)
        mean_probs = cls_probs.mean(axis=0)
        
        ax = axes[cls]
        colors_bar = [C_GRU if c == cls else '#333' for c in range(num_classes)]
        bars = ax.bar(range(num_classes), mean_probs, color=colors_bar, alpha=0.85)
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=60, ha='right', fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'"{class_names[cls]}"', fontsize=10, fontweight='bold')
        if cls == 0: ax.set_ylabel('Mean Softmax')
        ax.grid(True, axis='y', alpha=0.2)
        
        # Annotate the correct class confidence
        ax.text(cls, mean_probs[cls] + 0.02, f'{mean_probs[cls]*100:.0f}%', 
                ha='center', fontsize=8, color='#38bdf8', fontweight='bold')
    
    plt.suptitle(f'{name} — What the Model "Thinks" for Each Sign', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_softmax_profiles.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Softmax profiles saved for {name}")

# ═══════════════════════════════════════════════════════════
#  TEST 8: DECISION BOUNDARY — Interpolate between pairs
# ═══════════════════════════════════════════════════════════
def test_decision_boundary(model, X_test, y_test, class_names, name, out_dir):
    """Linearly interpolate between class centroids to find decision boundaries."""
    print(f"  [TEST 8] Decision Boundary Probing for {name}...")
    
    num_classes = len(class_names)
    centroids = []
    for cls in range(num_classes):
        mask = y_test == cls
        centroids.append(X_test[mask].mean(axis=0))
    
    # Pick interesting pairs: each class vs its nearest rival
    # Use all unique pairs
    pairs = [(i, j) for i in range(num_classes) for j in range(i+1, num_classes)]
    
    n_pairs = len(pairs)
    cols = min(7, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1: axes = [axes]
    axes_flat = [axes] if n_pairs == 1 else np.array(axes).flatten()
    
    for idx, (i, j) in enumerate(pairs):
        ax = axes_flat[idx]
        alphas = np.linspace(0, 1, 50)
        probs_i, probs_j = [], []
        
        for alpha in alphas:
            sample = (1 - alpha) * centroids[i] + alpha * centroids[j]
            pred = model.predict(sample[np.newaxis], verbose=0)[0]
            probs_i.append(pred[i])
            probs_j.append(pred[j])
        
        ax.plot(alphas, probs_i, '-', color=C_GRU, linewidth=2, label=class_names[i])
        ax.plot(alphas, probs_j, '-', color=C_LSTM, linewidth=2, label=class_names[j])
        ax.axhline(y=0.5, color='#666', linestyle=':', alpha=0.5)
        ax.set_title(f'{class_names[i]}↔{class_names[j]}', fontsize=8, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([class_names[i][:3], '50/50', class_names[j][:3]], fontsize=6)
        if idx == 0: ax.legend(fontsize=6)
    
    # Hide unused axes
    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle(f'{name} — Decision Boundaries Between Class Pairs', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name.lower().replace(" ","_")}_decision_boundaries.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Decision boundaries saved for {name}")

# ═══════════════════════════════════════════════════════════
#                        MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  SignVision — Deep Model Interpretability Analysis    ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    X = np.load(X_PATH); y = np.load(Y_PATH)
    with open(CLASSES_PATH) as f: class_map = json.load(f)
    class_names = [k for k,v in sorted(class_map.items(), key=lambda x: x[1])]
    num_classes = len(class_names)
    shape = (X.shape[1], X.shape[2])
    
    print(f"\n  Dataset: {X.shape[0]} samples, {num_classes} classes ({', '.join(class_names)})")
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_tr_c = to_categorical(y_tr, num_classes)
    y_te_c = to_categorical(y_te, num_classes)
    
    # Train
    gru_m, _ = train(build_conv_gru, "Conv1D+GRU", X_tr, y_tr_c, X_te, y_te_c, shape, num_classes)
    lstm_m, _ = train(build_bilstm, "Bi-LSTM", X_tr, y_tr_c, X_te, y_te_c, shape, num_classes)
    
    print(f"\n{'═'*55}")
    print(f"  Running 8 Deep Interpretability Tests...")
    print(f"{'═'*55}")
    
    # 1. Gradient Saliency + Body-Part Attribution
    test_gradient_saliency(gru_m, X_te, y_te, class_names, "Conv1D+GRU", OUT)
    test_gradient_saliency(lstm_m, X_te, y_te, class_names, "Bi-LSTM", OUT)
    
    # 2. Temporal Frame Importance
    test_temporal_importance(gru_m, X_te, y_te, class_names, "Conv1D+GRU", OUT)
    test_temporal_importance(lstm_m, X_te, y_te, class_names, "Bi-LSTM", OUT)
    
    # 3. Occlusion Sensitivity (body part masking)
    test_occlusion(gru_m, X_te, y_te_c, class_names, "Conv1D+GRU", OUT)
    test_occlusion(lstm_m, X_te, y_te_c, class_names, "Bi-LSTM", OUT)
    
    # 4. Layer-wise Feature Evolution
    test_layer_evolution(gru_m, X_te, y_te, class_names, "Conv1D+GRU", OUT)
    test_layer_evolution(lstm_m, X_te, y_te, class_names, "Bi-LSTM", OUT)
    
    # 5. Noise Robustness
    test_noise_robustness([gru_m, lstm_m], ["Conv1D+GRU", "Bi-LSTM"], X_te, y_te_c, OUT)
    
    # 6. Frame Dropout Robustness
    test_frame_dropout([gru_m, lstm_m], ["Conv1D+GRU", "Bi-LSTM"], X_te, y_te_c, OUT)
    
    # 7. Per-Class Softmax Profiles
    test_softmax_profiles(gru_m, X_te, y_te, class_names, "Conv1D+GRU", OUT)
    test_softmax_profiles(lstm_m, X_te, y_te, class_names, "Bi-LSTM", OUT)
    
    # 8. Decision Boundaries
    test_decision_boundary(gru_m, X_te, y_te, class_names, "Conv1D+GRU", OUT)
    test_decision_boundary(lstm_m, X_te, y_te, class_names, "Bi-LSTM", OUT)
    
    print(f"\n\n{'#'*55}")
    print(f"  DEEP ANALYSIS COMPLETE — {16} charts generated")
    print(f"  Saved to: {OUT}")
    print(f"{'#'*55}")

if __name__ == '__main__':
    main()
