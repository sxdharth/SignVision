import cv2
import numpy as np
import os
import json
import time # Added for cooldown
from tensorflow.keras.models import load_model
from feature_extractor import FeatureExtractor

import os
import os
# Define paths relative to this file's location (src/inference_engine.py)
# Root is one level up
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENERAL_MODEL_PATH = os.path.join(ROOT_DIR, 'Models', 'final_model.h5')
GENERAL_CLASSES_PATH = os.path.join(ROOT_DIR, 'Data', 'combined_classes.json')

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
        self.load_resources()

    def load_resources(self, mode='general'):
        if mode == 'iot':
            model_path = SMART_HOME_MODEL_PATH
            classes_path = SMART_HOME_CLASSES_PATH
        else:
            model_path = GENERAL_MODEL_PATH
            classes_path = GENERAL_CLASSES_PATH
            
        if os.path.exists(model_path):
            print(f"Loading {mode} model from {model_path}...")
            self.model = load_model(model_path)
        else:
            print(f"Model not found at {model_path}")
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
        self.sequence.append(landmarks)
        
        # Keep only the last MAX_LENGTH frames
        self.sequence = self.sequence[-MAX_LENGTH:]

        # Cooldown Check - After appending landmarks so buffer doesn't jump
        if time.time() - self.last_prediction_time < 0.5:
            return None, 0.0
        
        if len(self.sequence) == MAX_LENGTH:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            
            # Get top predictions
            top_indices = np.argsort(res)[::-1]
            
            for idx in top_indices:
                label = self.classes[idx]
                confidence = float(res[idx])

                # 1. Mode Filtering (Prioritize Mode Validity)
                mode = getattr(self, 'mode', 'general')
                if mode == 'spelling':
                    # In spelling mode, ignore words (length > 1) matches
                    if len(label) > 1:
                        continue
                elif mode == 'iot':
                    pass # Allow 'none' class to be predicted so UI knows we are idle
                        
                # Removed confidence threshold from the backend so the UI can always display
                # the top guess. The frontend will now handle the 30% execution threshold.
                print(f"Prediction: {label} ({confidence:.2f})") # Debug print
                self.last_prediction_time = time.time() # Update cooldown timer
                return label, confidence
            
            print(f"No valid prediction. Top was: {self.classes[top_indices[0]]} ({float(res[top_indices[0]]):.2f})")
            
        return None, 0.0


    @property
    def is_cooldown_active(self):
        return (time.time() - self.last_prediction_time) < 0.5

    def set_mode(self, mode):
        if getattr(self, 'mode', None) != mode:
            self.mode = mode
            self.load_resources(mode=mode)
            self.sequence = [] # Clear sequence cache on mode switch

    def close(self):
        self.extractor.close()
