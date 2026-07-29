import os, sys, json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, 'src'))

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA = os.path.join(ROOT, 'Data')
MODEL = os.path.join(ROOT, 'Models', 'video_call_model.h5')

X = np.load(os.path.join(DATA, 'X_video_call.npy'))
y = np.load(os.path.join(DATA, 'y_video_call.npy'))
classes = json.load(open(os.path.join(DATA, 'video_call_classes.json')))
inv = {v: k for k, v in classes.items()}

# Because 'no' has 0 samples, it might not be in the y array at all
# Let's find out which labels actually exist in the data
unique_y = np.unique(y)
names = [inv[i] for i in unique_y]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = load_model(MODEL)
preds = model.predict(X_test, verbose=0)
y_pred = np.argmax(preds, axis=1)

report = classification_report(y_test, y_pred, labels=unique_y, target_names=names, digits=2)

with open('ppt_metrics.txt', 'w') as f:
    f.write(report)
    
print("Metrics saved to ppt_metrics.txt!")
