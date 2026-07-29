import os, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Recreate the exact dataset split
X = np.load('Data/X_video_call.npy')
y = np.load('Data/y_video_call.npy')

np.random.seed(42) # Ensure deterministic augmentation for consistency if needed, though it was random originally.
X_aug, y_aug = [], []
for i in range(len(X)):
    seq, label = X[i], y[i]
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

X = np.array(X_aug)
y = np.array(y_aug)
y_cat = to_categorical(y).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
model = load_model('Models/video_call_model_gru.h5')
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)

with open('acc_out.txt', 'w', encoding='utf-8') as f:
    f.write(f"Train Accuracy: {train_acc*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
