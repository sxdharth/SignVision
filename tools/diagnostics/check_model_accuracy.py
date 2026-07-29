"""
Quick evaluation of the video call model — writes results to eval_output.py
"""
import os, sys, json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, 'src'))

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA = os.path.join(ROOT, 'Data')
MODEL = os.path.join(ROOT, 'Models', 'video_call_model.h5')

X = np.load(os.path.join(DATA, 'X_video_call.npy'))
y = np.load(os.path.join(DATA, 'y_video_call.npy'))
classes = json.load(open(os.path.join(DATA, 'video_call_classes.json')))
inv = {v: k for k, v in classes.items()}
names = [inv[i] for i in range(len(inv))]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = load_model(MODEL)
preds = model.predict(X_test, verbose=0)
y_pred = np.argmax(preds, axis=1)

report = classification_report(y_test, y_pred, target_names=names)
cm = confusion_matrix(y_test, y_pred)

overall = np.mean(y_pred == y_test) * 100

# Write as Python file for easy viewing
with open(os.path.join(ROOT, 'eval_output.py'), 'w', encoding='utf-8') as f:
    f.write(f'# Model Evaluation Results\n')
    f.write(f'# Data: {X.shape[0]} samples, {len(names)} classes\n')
    f.write(f'# Classes: {names}\n')
    f.write(f'# Test set: {len(X_test)} samples\n')
    f.write(f'# Overall Accuracy: {overall:.1f}%\n')
    f.write(f'#\n')
    f.write(f'# Classification Report:\n')
    for line in report.split('\n'):
        f.write(f'# {line}\n')
    f.write(f'#\n')
    f.write(f'# Confusion Matrix (rows=actual, cols=predicted):\n')
    f.write(f'# {"":>9s}  {"  ".join(f"{n:>8s}" for n in names)}\n')
    for i, row in enumerate(cm):
        f.write(f'# {names[i]:>9s}  {"  ".join(f"{v:>8d}" for v in row)}\n')
    f.write(f'#\n')
    f.write(f'# Confused Pairs:\n')
    found = False
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i][j] > 0:
                found = True
                f.write(f'#   {names[i]} -> {names[j]}: {cm[i][j]} ({cm[i][j]/cm[i].sum()*100:.0f}%)\n')
    if not found:
        f.write(f'#   None - model has 100% accuracy on test set\n')

print("Done: eval_output.py")
