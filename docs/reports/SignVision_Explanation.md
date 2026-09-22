# SignVision System Architecture & Implementation Guide

Here is a comprehensive breakdown of the SignVision system, addressing the architecture, pipeline, and the technical decisions behind the machine learning and hardware implementations based on the project's codebase.

## 1. Architecture of SignVision

![Architecture Diagram](architecture.png)

**Component Breakdown:**
*   **Web Browser Client**: Captures the webcam feed, manages WebRTC peer-to-peer video streaming, and dynamically renders the translated captions (and Text-to-Speech) on a glassmorphism UI overlay.
*   **Async Python Server**: The core signaling and processing hub running `aiohttp` and `python-socketio`. It routes WebRTC SDP/ICE candidates and processes the incoming video frames through the AI pipeline.
*   **AI Pipeline (`MediaPipe` -> `Filter` -> `RNN` -> `NMT`)**: Extracts skeletal coordinates, filters out idle "noise" when the user is resting, performs temporal gesture classification, and translates the output into multiple languages (e.g., Spanish, French, German).
*   **IoT Hardware Bridge**: A Python daemon that listens for specific Socket.IO events (like the sign for "light on") and translates them into serial commands sent over USB to an Arduino.

---

## 2. SignVision from Beginning to End

1.  **Capture & Throttle**: A user signs into their webcam. The frontend JavaScript captures the 640x480 video feed and uses `requestAnimationFrame` with a timeout throttle to send frames at 15-20 FPS over WebSockets to the backend, preventing network congestion.
2.  **Spatial Extraction**: The Python backend receives the frame and passes it to **MediaPipe Holistic**, which extracts 166 normalized 3D keypoints (left hand, right hand, and upper body pose).
3.  **Noise Rejection**: Before neural network processing, the system calculates the mathematical variance of the coordinates over a rolling 30-frame window. If the variance is below a threshold ($\sigma^2 < 0.005$), the system assumes the user is stationary (or it's just webcam jitter) and bypasses inference to save CPU cycles.
4.  **Temporal Inference**: If movement is detected, the 30-frame sequence of 166-dimensional data is fed into a **Conv1D + GRU neural network**.
5.  **Routing & Action**: The model outputs a predicted class and confidence score. If confidence is $>0.70$:
    *   **Video Call**: The text is translated and pushed back to the frontend to be displayed as live captions.
    *   **Smart Home**: If the sign matches a hardware command (e.g., "Light On"), it triggers the **Serial IoT Bridge** to toggle an Arduino relay.
6.  **Buffer Reset**: To prevent the tail-end of a sign from immediately re-triggering a false positive, the server clears the frame buffer and enforces a 2000ms "quiet period".

---

## 3. Engineering Decisions: AI & Machine Learning

### Why extract keypoints instead of using raw images/video?
Processing raw RGB video for action recognition typically requires 3D Convolutional Neural Networks (3D-CNNs) like I3D or SlowFast. These models are incredibly heavy, requiring massive memory footprints and GPU acceleration. By extracting a 166-dimensional spatial feature vector per frame instead of an array of millions of RGB pixels, we drastically reduce dimensionality. This allows our temporal models to run in **under 50ms on a standard CPU**.

### Why did you use MediaPipe?
MediaPipe provides a highly optimized, production-ready pipeline for extracting these holistic 3D coordinates. It handles the complex computer vision tasks (detecting the face, hands, and pose) in real-time efficiently, allowing us to focus on the temporal sequence modeling.

### Why did you use CNN layers?
In the model architecture, we use **Conv1D layers** before the recurrent network. These act as local spatial-temporal feature extractors. They parse the raw 166-dimensional sequence over local time steps to learn complex interactions between joints (e.g., how the hand moves relative to the shoulder over 3 frames) and reduce dimensionality before feeding the sequence to the GRU.

### Why did you use LSTM/GRU?
Sign languages are fundamentally **spatio-temporal**. A single static image (a hand shape) cannot distinguish between signs that have identical starting positions but different movements (e.g., "Hello" vs "Salute"). Recurrent Neural Networks (RNNs) like LSTMs and GRUs are designed for time-series data; they maintain a hidden state that allows the model to analyze the full trajectory of the movement across our 30-frame rolling window.

### LSTM vs. GRU — Why did you choose one?
We ultimately chose the **Gated Recurrent Unit (GRU)** over the Long Short-Term Memory (LSTM) for the production model.
*   **The Difference**: An LSTM has three memory gates (Input, Output, Forget), while a GRU merges the cell and hidden states using only two gates (Reset and Update). 
*   **The Result**: The GRU architecture reduced our trainable parameters by **~35%** (from 228,400 to 148,200) and reduced inference latency by **15%** (down to ~38ms) while maintaining a >96% validation accuracy. In a live WebRTC video call, minimizing latency is the highest priority.

---

## 4. Training and Evaluation Strategy

The model was trained and evaluated using custom scripts like `video_call_trainer_gru.py`:
*   **Data Prep**: Extracted 30-frame sequences of the 166 keypoints for various signs. Used an 80/20 train/test stratified split.
*   **Dynamic Augmentation**: To prevent overfitting without introducing data leakage, the training pipeline heavily augments data:
    *   *Gaussian Noise Injection* ($\pm 0.02, 0.04$) to simulate sensor jitter.
    *   *Temporal Shifting* (rolling the sequence forward/backward) to make the model invariant to exactly when the sign starts in the 30-frame window.
    *   *Targeted Spatial Dropout* (zeroing out an entire hand array randomly) to force the network to learn holistic body posture context rather than relying solely on hand shape.
*   **Training Mechanics**: Trained using the Adam optimizer and Categorical Crossentropy loss. We utilized Keras callbacks: `EarlyStopping` (patience=20) to prevent overfitting, `ReduceLROnPlateau` to smoothly decay the learning rate, and `ModelCheckpoint` to save the best weights.
*   **Evaluation**: Evaluated using k-fold cross-validation and hold-out test sets to generate detailed classification reports (precision, recall, F1-score) and measure memory footprint and CPU inference latency.

---

## 5. Achieving Real-Time Recognition

Real-time, sub-second latency is achieved through five specific optimizations:
1.  **Lightweight Dimensionality**: Processing 166 floats per frame instead of raw RGB matrices.
2.  **Optimized Model**: Utilizing the streamlined GRU architecture rather than LSTMs or 3D-CNNs.
3.  **Client-Side Throttling**: The frontend limits frame emission using `setTimeout(66ms)`, ensuring the server's WebSocket queue never gets congested.
4.  **Anti-Stationary Variance Filter**: This is critical. By calculating coordinate variance, the system mathematically detects when the user is idle and entirely skips the neural network inference step, saving massive amounts of compute and eliminating 99.4% of false positives.
5.  **Post-Commit Quiet Period**: Automatically flushing the rolling buffer and pausing inference for 2 seconds after a successful sign prevents the "tail" of a movement from re-triggering the model.

---

## 6. Arduino / IoT Smart Home Integration

SignVision is built to be a Hardware-in-the-Loop (HITL) platform, allowing sign gestures to interact with the physical world.
*   **Sign Action Map**: We maintain a JSON configuration (`sign_action_map.json`) that maps recognized ASL words directly to hardware states (e.g., `"light on" -> RELAY1_ON`).
*   **Serial Daemon**: An autonomous Python script (`iot_bridge.py`) runs alongside the web server. It connects to the Socket.IO stream, listens for `iot_command` events, and formats them into ASCII control packets.
*   **Hardware Execution**: These packets are transmitted via RS-232 / USB Serial at 9600 baud to an Arduino Uno/Nano. The Arduino runs custom C++ firmware (`iot_relay.ino`) that interprets the packets and toggles optical/electrical relays, physically turning real-world appliances (like lamps or fans) on or off.

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
