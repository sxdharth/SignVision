import os, sys, json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA = 'Data'
MODEL = 'Models/video_call_model_gru.h5'

X = np.load(os.path.join(DATA, 'X_video_call.npy'))
y = np.load(os.path.join(DATA, 'y_video_call.npy'))
classes = json.load(open(os.path.join(DATA, 'video_call_classes.json')))
inv = {v: k for k, v in classes.items()}
names = [inv[i] for i in range(len(inv))]

# Uses the same split as check_model_accuracy.py
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = load_model(MODEL)
preds = model.predict(X_test, verbose=0)
y_pred = np.argmax(preds, axis=1)
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test = np.argmax(y_test, axis=1)

report = classification_report(y_test, y_pred, target_names=names)
overall = np.mean(y_pred == y_test) * 100

with open('gru_eval_out.txt', 'w', encoding='utf-8') as f:
    f.write(f'Overall Accuracy: {overall:.2f}%\n\n')
    f.write(f'Classification Report:\n{report}\n')
