import cv2
import mediapipe as mp
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, frame):
        """
        Extracts landmarks from a frame using MediaPipe Holistic.
        Returns a flattened numpy array of landmarks (Pose + Left Hand + Right Hand).
        """
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image and find landmarks
        results = self.holistic.process(image)
        
        # Extract Pose landmarks
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33*3)
            
        # Extract Left Hand landmarks
        if results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        else:
            lh = np.zeros(21*3)
            
        # Extract Right Hand landmarks
        if results.right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        else:
            rh = np.zeros(21*3)
            
        # Local Normalization Function
        def normalize_set(landmarks, origin_index=0):
            if np.all(landmarks == 0):
                return landmarks
            
            # Reshape to (N, 3)
            points = landmarks.reshape(-1, 3)
            
            # Origin is the specified landmark (e.g., Nose or Wrist)
            origin = points[origin_index]
            
            # Center points
            centered = points - origin
            
            # Scale (max distance from origin)
            max_dist = np.max(np.abs(centered))
            if max_dist > 0:
                normalized = centered / max_dist
            else:
                normalized = centered
                
            return normalized.flatten()

        # Normalize each component locally
        pose = normalize_set(pose, origin_index=0) # Nose
        lh = normalize_set(lh, origin_index=0)     # Wrist
        rh = normalize_set(rh, origin_index=0)     # Wrist
            
        # Concatenate all landmarks
        return np.concatenate([pose, lh, rh])

    def close(self):
        self.holistic.close()
