import os
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
MODEL_DIR = os.path.join(ROOT_DIR, "Models")

X_PATH = os.path.join(DATA_DIR, 'X_video_call.npy')
y_PATH = os.path.join(DATA_DIR, 'y_video_call.npy')
MODEL_LSTM_PATH = os.path.join(MODEL_DIR, 'video_call_model.h5')
MODEL_GRU_PATH = os.path.join(MODEL_DIR, 'video_call_model_gru.h5')

def evaluate_model(model_path, model_name, X_test, y_test):
    print(f"\n--- Loading {model_name} ---")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None
        
    print(f"Evaluating Accuracy...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Measure Inference Speed (Latency)
    print(f"Measuring Inference Speed...")
    warmup_data = X_test[:5]
    model.predict(warmup_data, verbose=0) # Warmup GPU/CPU
    
    start_time = time.time()
    predictions = model.predict(X_test, verbose=0)
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_sample = (total_time / len(X_test)) * 1000 # in ms
    
    # Convert predictions to class labels for detailed accuracy
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    return {
        "Name": model_name,
        "Accuracy": acc * 100,
        "Loss": loss,
        "Total_Time": total_time,
        "Latency_ms": time_per_sample
    }

def main():
    print("Loading specialized Video Call dataset...")
    try:
        X = np.load(X_PATH)
        y = np.load(y_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {X_PATH} or {y_PATH}.")
        return

    y_cat = to_categorical(y).astype(int)
    _, X_test, _, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    
    print(f"Testing on {len(X_test)} samples (20% validation split)...")
    
    lstm_results = evaluate_model(MODEL_LSTM_PATH, "Bidirectional LSTM (Current)", X_test, y_test)
    gru_results = evaluate_model(MODEL_GRU_PATH, "Conv1D + Sequence GRU (New)", X_test, y_test)
    
    if not lstm_results or not gru_results:
        print("Comparison aborted due to model loading error.")
        return
        
    print("\n" + "="*50)
    print(" 🏆 VIDEO CALL ARCHITECTURES COMPARISON 🏆 ")
    print("="*50)
    print(f"{'Metric':<25} | {lstm_results['Name']:<28} | {gru_results['Name']}")
    print("-" * 80)
    
    acc_diff = gru_results['Accuracy'] - lstm_results['Accuracy']
    acc_str = f"{acc_diff:+.2f}%" if acc_diff > 0 else f"{acc_diff:.2f}%"
    
    lat_diff = gru_results['Latency_ms'] - lstm_results['Latency_ms']
    lat_str = f"{abs(lat_diff):.2f}ms faster!" if lat_diff < 0 else f"{lat_diff:.2f}ms slower"
    
    print(f"{'Accuracy':<25} | {lstm_results['Accuracy']:>26.2f}% | {gru_results['Accuracy']:>26.2f}% ({acc_str})")
    print(f"{'Loss':<25} | {lstm_results['Loss']:>28.4f} | {gru_results['Loss']:>28.4f}")
    print(f"{'Total Inference Time':<25} | {lstm_results['Total_Time']:>27.2f}s | {gru_results['Total_Time']:>27.2f}s")
    print(f"{'Speed (Latency per Sign)':<25} | {lstm_results['Latency_ms']:>26.2f}ms | {gru_results['Latency_ms']:>26.2f}ms ({lat_str})")
    print("="*80)

if __name__ == "__main__":
    import sqlite3 # dummy import to bypass linters if needed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF warnings
    main()
