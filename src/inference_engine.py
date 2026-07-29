import cv2
import numpy as np
import os
import json
import time # Added for cooldown
from tensorflow.keras.models import load_model
from feature_extractor import FeatureExtractor



# Define paths relative to this file's location (src/inference_engine.py)
# Root is one level up
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENERAL_MODEL_PATH = os.path.join(ROOT_DIR, 'Models', 'video_call_model.h5')
GENERAL_MODEL_GRU_PATH = os.path.join(ROOT_DIR, 'Models', 'video_call_model_gru.h5')
GENERAL_CLASSES_PATH = os.path.join(ROOT_DIR, 'Data', 'video_call_classes.json')

SMART_HOME_MODEL_PATH = os.path.join(ROOT_DIR, 'Models', 'smart_home_model.h5')
SMART_HOME_CLASSES_PATH = os.path.join(ROOT_DIR, 'Data', 'smart_home_classes.json')

MAX_LENGTH = 30


class InferenceEngine:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.sequence = []
        self.model = None
        self.classes = []
        self.last_prediction_time = 0 # Added for cooldown
        self.architecture = 'gru' # Default active architecture
        self.load_resources()

    def load_resources(self, mode='general'):
        if mode == 'iot':
            model_path = SMART_HOME_MODEL_PATH
            classes_path = SMART_HOME_CLASSES_PATH
        else:
            if self.architecture == 'gru':
                model_path = GENERAL_MODEL_GRU_PATH
            else:
                model_path = GENERAL_MODEL_PATH
            classes_path = GENERAL_CLASSES_PATH
            
        try:
            if os.path.exists(model_path):
                print(f"Loading {mode} model ({self.architecture}) from {model_path}...")
                self.model = load_model(model_path)
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            if self.architecture == 'gru' and mode == 'general':
                print(f"Attempting fast fallback to LSTM model...")
                self.architecture = 'lstm'
                self.load_resources(mode=mode)
                return
            else:
                self.model = None

        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                class_map = json.load(f)
                # Invert map to get index -> name
                max_idx = max(class_map.values()) if class_map else 0
                self.classes = [None] * (max_idx + 1)
                for name, idx in class_map.items():
                    self.classes[idx] = name
            print(f"Loaded {len(self.classes)} {mode} classes.")
        else:
            print(f"Classes file not found at {classes_path}")
            self.classes = []

    def predict(self, frame):
        """
        Processes a frame and returns (prediction, confidence).
        Returns (None, 0.0) if model is not loaded or sequence is too short.
        """
        if self.model is None:
            return None, 0.0

        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            if current_time - self.last_frame_time > 2.0:
                self.sequence = [] # Clear stale sequence buffer
        self.last_frame_time = current_time

        landmarks = self.extractor.extract_landmarks(frame)
        
        # Smart Tracker: Wipe sequence if hands disappear for 5 frames
        if np.all(landmarks[99:] == 0):
            self.missing_frames_count = getattr(self, 'missing_frames_count', 0) + 1
        else:
            self.missing_frames_count = 0
            
        if getattr(self, 'missing_frames_count', 0) >= 5:
            # Hands definitively dropped.
            # If the user completed a rapid sign in just a few frames, predictably evaluate it before wiping!
            if len(self.sequence) > 3 and self.model is not None:
                padded_seq = np.zeros((MAX_LENGTH, self.sequence[0].shape[0]), dtype='float32')
                padded_seq[:len(self.sequence)] = np.array(self.sequence)
                res = self.model.predict(np.expand_dims(padded_seq, axis=0), verbose=0)[0]
                
                top_indices = np.argsort(res)[::-1]
                for idx in top_indices:
                    label = self.classes[idx]
                    confidence = float(res[idx])
                    
                    if label is None: continue
                    mode = getattr(self, 'mode', 'general')
                    if mode == 'spelling' and len(label) > 1: continue
                        
                    if confidence >= 0.60:
                        print(f"DEBUG Early Prediction: {label} (Conf: {confidence:.2f})", flush=True)
                        self.last_prediction_time = current_time
                        self.sequence = []
                        return label, confidence
                    else:
                        print(f"DEBUG Early Eval dropped (Conf: {confidence:.2f} < 0.40)", flush=True)

            self.sequence = []
            return None, 0.0

        self.sequence.append(landmarks)
        
        # Keep only the last MAX_LENGTH frames
        self.sequence = self.sequence[-MAX_LENGTH:]

        if len(self.sequence) == MAX_LENGTH:
            # We explicitly removed all manual variance/wiper checks here because 
            # MediaPipe dynamically drops bodily landmarks based on webcam zoom,
            # which caused mathematically 0.0 variance and silently wiped valid signs!
            # The AI's 0.40 confidence gate accurately ignores noise natively!
            
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            
            # Get top predictions
            top_indices = np.argsort(res)[::-1]
            
            for idx in top_indices:
                label = self.classes[idx]
                confidence = float(res[idx])

                # Guard: skip any class that failed to load
                if label is None:
                    continue

                # 1. Mode Filtering
                mode = getattr(self, 'mode', 'general')
                if mode == 'spelling':
                    if len(label) > 1:
                        continue
                elif mode == 'iot':
                    pass
                        
                # Python Floor
                if confidence < 0.60:
                    continue
                    
                print(f"DEBUG Prediction: {label} (Conf: {confidence:.2f})", flush=True) 
                self.last_prediction_time = time.time()
                self.sequence = [] # Wipe after successful deep peak prediction
                return label, confidence
            
            print(f"DEBUG No valid prediction. Top was: {self.classes[top_indices[0]]} ({float(res[top_indices[0]]):.2f})", flush=True)
            
        return None, 0.0

    def predict_batch(self, frames):
        """
        Processes a batch of frames using a SLIDING WINDOW approach.
        
        1. Extract landmarks from ALL frames
        2. Slide a 30-frame window across them (step=5 for speed)
        3. Run the model on each window
        4. Return the prediction with the HIGHEST confidence
        
        This catches the sign no matter when during the recording it was performed.
        """
        if self.model is None:
            return None, 0.0

        if not frames or len(frames) == 0:
            return None, 0.0

        # Step 1: Extract landmarks from all frames
        print(f"Extracting landmarks from {len(frames)} frames...", flush=True)
        landmarks_list = []
        for frame in frames:
            landmarks = self.extractor.extract_landmarks(frame)
            landmarks_list.append(landmarks)

        if len(landmarks_list) == 0:
            return None, 0.0

        total_frames = len(landmarks_list)
        print(f"Landmarks extracted. Total: {total_frames} frames", flush=True)

        # Step 2: If fewer than MAX_LENGTH frames, pad and run once
        if total_frames < MAX_LENGTH:
            padded = np.zeros((MAX_LENGTH, landmarks_list[0].shape[0]), dtype='float32')
            padded[-total_frames:] = np.array(landmarks_list)
            windows = [padded]
        else:
            # step=15 gives 3 overlapping windows across 60 frames:
            #   Window 1: [0:30]  → first 2 seconds
            #   Window 2: [15:45] → middle 2 seconds  ← catches mid-recording signs
            #   Window 3: [30:60] → last 2 seconds
            step = 15
            windows = []
            last_start = -1
            for start in range(0, total_frames - MAX_LENGTH + 1, step):
                window = np.array(landmarks_list[start:start + MAX_LENGTH], dtype='float32')
                windows.append(window)
                last_start = start

            # Always include the very last window if not already included
            final_start = total_frames - MAX_LENGTH
            if final_start != last_start:
                last_window = np.array(landmarks_list[final_start:], dtype='float32')
                windows.append(last_window)

        print(f"Running model on {len(windows)} sliding windows...", flush=True)

        # Step 3: Run model on all windows, track best prediction
        best_label = None
        best_confidence = 0.0
        mode = getattr(self, 'mode', 'general')

        # Batch predict all windows at once for speed
        batch_input = np.array(windows, dtype='float32')
        all_results = self.model.predict(batch_input, verbose=0)

        for i, res in enumerate(all_results):
            top_indices = np.argsort(res)[::-1]

            for idx in top_indices:
                label = self.classes[idx]
                confidence = float(res[idx])

                if label is None:
                    continue
                if mode == 'spelling' and len(label) > 1:
                    continue
                if confidence < 0.75:  # Strict threshold to avoid false positives
                    break  # Sorted descending — nothing below this will pass

                # This is the best valid label for this window
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_label = label
                    print(f"  Window {i}: {label} ({confidence:.2f}) <- new best", flush=True)
                break  # Only need the top valid label per window

        if best_label:
            print(f"DEBUG Sliding Window Best: {best_label} (Conf: {best_confidence:.2f})", flush=True)
            self.last_prediction_time = time.time()
            return best_label, best_confidence

        print(f"DEBUG Sliding Window: No valid prediction across {len(windows)} windows", flush=True)
        return None, 0.0


    @property
    def is_cooldown_active(self):
        return (time.time() - self.last_prediction_time) < 0.5

    def set_mode(self, mode):
        if getattr(self, 'mode', None) != mode:
            self.mode = mode
            self.load_resources(mode=mode)
            self.sequence = [] # Clear sequence cache on mode switch
            
    def set_architecture(self, arch):
        if self.architecture != arch:
            self.architecture = arch
            self.load_resources(mode=getattr(self, 'mode', 'general'))
            self.sequence = [] # Clear sequence buffer to avoid dim errors
            
    def clear_sequence(self):
        """Wipes the LSTM frame buffer so it doesn't predict on stale data."""
        self.sequence = []

    def close(self):
        self.extractor.close()
