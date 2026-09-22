content = """
---

## 7. Problems Faced & Solutions

While building SignVision, several critical engineering challenges were encountered and solved:

### 1. Model Overfitting (100% Test Accuracy, but Failed Live)
*   **Problem:** The model achieved 100% accuracy on the test data but misclassified signs in the live video call. It had memorized the training data and couldn't generalize to the slight differences caused by JPEG compression noise, MediaPipe landmark variance, and temporal frame jitter in the live pipeline.
*   **Solution:** Implemented **5x dynamic data augmentation** during training (without data leakage), including Gaussian noise injection, temporal shifting, and targeted spatial dropout. Also increased dropout rates and L2 regularization to force generalization.

### 2. Constant "Hello" False Positives on Idle Frames
*   **Problem:** The generalized model constantly predicted "hello" when the user was sitting still. The training data didn't contain an "idle" class, and sensor noise/autofocus jitter bypassed basic zero-motion strict checks.
*   **Solution:** Designed an **Anti-Stationary Variance Filter** that calculates the mathematical variance of landmark movement across the rolling 30-frame sequence. If the variance is below `0.005`, the system mathematically ignores the webcam jitter and bypasses inference entirely.

### 3. Server Crashes and WebSocket Congestion
*   **Problem:** Using `setInterval` on the frontend to fire frames at 30 FPS caused a massive queue of pending bytes to pile up in memory because the backend inference took ~50-100ms per frame under heavy load. This caused the `aiohttp` backend to crash with an `OSError`.
*   **Solution:** Switched the frontend camera loop to an asynchronous `requestAnimationFrame` wait pattern with a `setTimeout(66ms)` delay. This guarantees client-side throttling to ~15-20 FPS, matching the server's processing limits and preventing network buffer crashes.

### 4. Buffer Ghosting (Phantom Predictions)
*   **Problem:** After a sign was successfully recognized and committed, the backend LSTM buffer still contained old frames. On unlock, it immediately re-predicted the same (or wrong) sign from the leftover data.
*   **Solution:** Engineered a **Post-Commit Buffer Reset**. After any sign is committed, the backend actively clears the 30-frame buffer and enforces a strict 2000ms quiet period where all new frames are ignored, allowing the user to reset their hands.

### 5. Frame Resolution and Quality Mismatch
*   **Problem:** The training dataset was built using raw 640x480 resolution frames, but the live WebRTC inference pipeline was downscaling frames to 1/4 resolution with 50% JPEG quality to save bandwidth, drastically degrading MediaPipe accuracy.
*   **Solution:** Standardized the pipeline so that live inference **must match** training conditions exactly, sending frames at 640x480 with 80% JPEG quality over the WebSocket.
"""

with open("SignVision_Explanation.md", "a", encoding="utf-8") as f:
    f.write(content)
