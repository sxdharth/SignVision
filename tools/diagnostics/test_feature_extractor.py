import cv2
import numpy as np
from feature_extractor import FeatureExtractor

def test():
    extractor = FeatureExtractor()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = extractor.extract_landmarks(img)
    print(f"Landmarks shape: {landmarks.shape}")
    # 33*3 (Pose) + 21*3 (Left Hand) + 21*3 (Right Hand) = 99 + 63 + 63 = 225
    expected_shape = (225,)
    if landmarks.shape == expected_shape:
        print("Test passed!")
    else:
        print(f"Test failed! Expected {expected_shape}, got {landmarks.shape}")
    extractor.close()

if __name__ == "__main__":
    test()
