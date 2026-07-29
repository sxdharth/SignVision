import numpy as np
import os
from tensorflow.keras.models import load_model

ROOT_DIR = 'd:/SignVision_S8_V2'
MODEL_PATH = os.path.join(ROOT_DIR, 'Models', 'smart_home_model.h5')
X_PATH = os.path.join(ROOT_DIR, 'Data', 'X_smart_home.npy')
Y_PATH = os.path.join(ROOT_DIR, 'Data', 'y_smart_home.npy')

print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading data...")
X = np.load(X_PATH)
y = np.load(Y_PATH)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Evaluate
print("Evaluating model on training data...")
predictions = model.predict(X, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
true_classes = y

# Calculate overall accuracy
correct = np.sum(pred_classes == true_classes)
print(f"Overall Accuracy: {correct}/{len(y)} ({correct/len(y)*100:.2f}%)")

for class_idx in np.unique(y):
    mask = (true_classes == class_idx)
    total = np.sum(mask)
    if total > 0:
        correct = np.sum(pred_classes[mask] == class_idx)
        print(f"Class {class_idx}: {correct}/{total} correct ({correct/total*100:.2f}%)")
    else:
        print(f"Class {class_idx}: 0 samples")
