# SignVision: Engineering Architecture & Implementation Report

---

> **An end-to-end distributed system designed for real-time sign language recognition, translation, bi-directional video calling, and hardware IoT automation.**

## 1. System Architecture

![Architecture Diagram](architecture.png)

### Core Components
*   **Web Browser Client:** Captures the webcam feed, manages WebRTC peer-to-peer video streaming, and dynamically renders the translated captions (and Text-to-Speech) on a glassmorphism UI overlay.
*   **Async Python Server:** The core signaling and processing hub running `aiohttp` and `python-socketio`. It routes WebRTC SDP/ICE candidates and processes incoming video frames through the AI pipeline.
*   **AI Pipeline (`MediaPipe` &rarr; `Variance Filter` &rarr; `GRU` &rarr; `NMT`):** Extracts skeletal coordinates, mathematically filters out idle "noise" when the user is resting, performs temporal gesture classification, and translates the output into multiple languages.
*   **IoT Hardware Bridge:** A Python daemon that listens for specific Socket.IO events (like the sign for "light on") and translates them into serial commands sent over USB to an Arduino.

---

## 2. The End-to-End Pipeline

1.  **Capture & Throttle:** A user signs into their webcam. The frontend JavaScript captures the `640x480` video feed and uses an asynchronous `requestAnimationFrame` loop with a timeout throttle to send frames at **15-20 FPS** over WebSockets to the backend, preventing network congestion.
2.  **Spatial Extraction:** The Python backend receives the frame and passes it to **MediaPipe Holistic**, which extracts exactly **166 normalized 3D keypoints** (left hand, right hand, and upper body pose).
3.  **Noise Rejection:** Before neural network processing, the system calculates the mathematical variance of the coordinates over a rolling 30-frame window. If the variance is below the strict threshold ($&sigma;^2 < 0.005$), the system assumes the user is stationary and bypasses inference to save CPU cycles.
4.  **Temporal Inference:** If movement is detected, the 30-frame sequence of 166-dimensional data is fed into a **Conv1D + GRU** neural network.
5.  **Routing & Action:** The model outputs a predicted class and confidence score. If confidence $> 0.70$:
    *   *Video Call:* The text is translated and pushed back to the frontend to be displayed as live captions.
    *   *Smart Home:* If the sign matches a hardware command, it triggers the **Serial IoT Bridge** to toggle an Arduino relay.
6.  **Buffer Reset:** To prevent the tail-end of a sign from immediately re-triggering a false positive, the server clears the frame buffer and enforces a **2000ms quiet period**.

---

## 3. Engineering Decisions: AI & Machine Learning

### Why extract keypoints instead of using raw video?
Processing raw RGB video for action recognition typically requires 3D Convolutional Neural Networks (3D-CNNs) like I3D. These models require massive memory footprints and GPU acceleration. By extracting a 166-dimensional spatial feature vector per frame instead of millions of RGB pixels, we drastically reduce dimensionality. This allows our temporal models to run in **under 50ms on a standard CPU**.

### Why MediaPipe?
MediaPipe provides a highly optimized, production-ready pipeline for extracting holistic 3D coordinates. It handles the complex computer vision tasks (detecting the face, hands, and pose) efficiently in real-time, allowing us to focus entirely on temporal sequence modeling.

### Why Conv1D Layers?
We use **Conv1D layers** before the recurrent network. These act as local spatial-temporal feature extractors. They parse the raw 166-dimensional sequence over local time steps to learn complex interactions between joints (e.g., how the hand moves relative to the shoulder over 3 frames) and reduce dimensionality before feeding the sequence to the GRU.

### Why LSTM / GRU?
Sign languages are fundamentally **spatio-temporal**. A single static image cannot distinguish between signs that have identical starting positions but different movements (e.g., *"Hello"* vs *"Salute"*). Recurrent Neural Networks (RNNs) like LSTMs and GRUs maintain a hidden state that allows the model to analyze the full trajectory of the movement across our 30-frame rolling window.

### LSTM vs. GRU — Why did we choose GRU?
We ultimately chose the **Gated Recurrent Unit (GRU)** over the Long Short-Term Memory (LSTM) for the production model.
> **The Result:** The GRU architecture reduced our trainable parameters by **~35%** (from 228k to 148k) and reduced inference latency by **15%** (down to ~38ms) while maintaining a >96% validation accuracy. In a live WebRTC video call, minimizing latency is the highest priority.

---

## 4. Training and Evaluation Strategy

The model was trained and evaluated using custom data pipelines:
*   **Data Preparation:** Extracted 30-frame sequences of the 166 keypoints for various signs. Used an 80/20 train/test stratified split.
*   **Dynamic Augmentation:** To prevent overfitting without introducing data leakage, the pipeline heavily augments data:
    *   *Gaussian Noise Injection* ($\pm 0.02, 0.04$) to simulate sensor jitter.
    *   *Temporal Shifting* (rolling the sequence forward/backward) to make the model invariant to exactly when the sign starts.
    *   *Targeted Spatial Dropout* (zeroing out an entire hand array randomly) to force the network to learn holistic body posture context.
*   **Training Mechanics:** Trained using the Adam optimizer and Categorical Crossentropy loss. We utilized Keras callbacks: `EarlyStopping` (patience=20) to prevent overfitting, `ReduceLROnPlateau` to smoothly decay the learning rate, and `ModelCheckpoint` to save the best weights.
*   **Evaluation:** Evaluated using k-fold cross-validation and hold-out test sets to generate detailed classification reports (precision, recall, F1-score) and measure memory footprint and CPU inference latency.

---

## 5. Achieving Real-Time Recognition

Real-time, sub-second latency is achieved through five specific optimizations:
1.  **Lightweight Dimensionality:** Processing 166 floats per frame instead of raw RGB matrices.
2.  **Optimized Model:** Utilizing the streamlined GRU architecture rather than heavy LSTMs or 3D-CNNs.
3.  **Client-Side Throttling:** The frontend limits frame emission using `setTimeout(66ms)`, ensuring the server's WebSocket queue never gets congested.
4.  **Anti-Stationary Variance Filter:** By calculating coordinate variance, the system mathematically detects when the user is idle and entirely skips the neural network inference step, saving massive compute and eliminating 99.4% of false positives.
5.  **Post-Commit Quiet Period:** Automatically flushing the rolling buffer and pausing inference for 2 seconds after a successful sign prevents the "tail" of a movement from re-triggering the model.

---

## 6. Arduino / IoT Smart Home Integration

SignVision is built to be a Hardware-in-the-Loop (HITL) platform, allowing sign gestures to interact with the physical world.
*   **Sign Action Map:** A JSON configuration (`sign_action_map.json`) maps recognized ASL words directly to hardware states (e.g., `"light on" &rarr; RELAY1_ON`).
*   **Serial Daemon:** An autonomous Python script (`iot_bridge.py`) connects to the Socket.IO stream, listens for `iot_command` events, and formats them into ASCII control packets.
*   **Hardware Execution:** These packets are transmitted via RS-232 / USB Serial at 9600 baud to an Arduino Uno/Nano. The Arduino runs custom C++ firmware (`iot_relay.ino`) that interprets the packets and toggles optical relays, physically switching real-world appliances.

---

## 7. Critical Challenges & Engineering Solutions

While building SignVision, several critical engineering challenges were encountered and solved:

### Challenge 1: Model Overfitting (Failed Live)
*   **Problem:** The model achieved 100% accuracy on the test data but misclassified signs in the live video call. It had memorized the training data and couldn't generalize to the slight differences caused by JPEG compression noise, MediaPipe landmark variance, and temporal frame jitter in the live pipeline.
*   **Solution:** Implemented **5x dynamic data augmentation** during training (without data leakage), including Gaussian noise injection, temporal shifting, and targeted spatial dropout. Also increased dropout rates and L2 regularization to force generalization.

### Challenge 2: Constant "Hello" False Positives on Idle Frames
*   **Problem:** The generalized model constantly predicted "hello" when the user was sitting still. The training data didn't contain an "idle" class, and sensor noise/autofocus jitter bypassed basic zero-motion strict checks.
*   **Solution:** Designed an **Anti-Stationary Variance Filter** that calculates the mathematical variance of landmark movement across the rolling 30-frame sequence. If the variance is below `0.005`, the system mathematically ignores the webcam jitter and bypasses inference entirely.

### Challenge 3: Server Crashes and WebSocket Congestion
*   **Problem:** Using `setInterval` on the frontend to fire frames at 30 FPS caused a massive queue of pending bytes to pile up in memory because the backend inference took ~50-100ms per frame under heavy load. This caused the `aiohttp` backend to crash with an `OSError`.
*   **Solution:** Switched the frontend camera loop to an asynchronous `requestAnimationFrame` wait pattern with a `setTimeout(66ms)` delay. This guarantees client-side throttling to ~15-20 FPS, matching the server's processing limits and preventing network buffer crashes.

### Challenge 4: Buffer Ghosting (Phantom Predictions)
*   **Problem:** After a sign was successfully recognized and committed, the backend LSTM buffer still contained old frames. On unlock, it immediately re-predicted the same (or wrong) sign from the leftover data.
*   **Solution:** Engineered a **Post-Commit Buffer Reset**. After any sign is committed, the backend actively clears the 30-frame buffer and enforces a strict 2000ms quiet period where all new frames are ignored, allowing the user to reset their hands.

### Challenge 5: Frame Resolution and Quality Mismatch
*   **Problem:** The training dataset was built using raw 640x480 resolution frames, but the live WebRTC inference pipeline was downscaling frames to 1/4 resolution with 50% JPEG quality to save bandwidth, drastically degrading MediaPipe accuracy.
*   **Solution:** Standardized the pipeline so that live inference **must match** training conditions exactly, sending frames at 640x480 with 80% JPEG quality over the WebSocket.

---

## 8. Technology Stack & Rationale

*   **MediaPipe Holistic:** Used to extract 166 3D landmarks in real-time. Chosen because it provides state-of-the-art hand and pose tracking without requiring depth cameras, replacing computationally heavy pixel-level processing.
*   **TensorFlow / Keras:** Used to construct and train the **Conv1D + GRU** neural network. Chosen for its robust support for sequential time-series modeling and easy edge deployment.
*   **Python (AIOHTTP + Socket.IO):** Used for the backend server. Chosen because it natively supports asynchronous I/O, allowing the server to handle WebRTC signaling and high-frequency WebSocket frame streaming without blocking threads.
*   **HTML5 / WebRTC / Vanilla JS:** Used for the frontend client. Chosen for native P2P video streaming and sub-millisecond DOM updates without the overhead of heavy web frameworks.
*   **PySerial & Arduino C++:** Used for the Smart Home IoT bridge. Chosen for reliable, low-level RS-232 serial communication between the AI inference server and physical hardware relays.

---

## 9. The Gesture Lifecycle: From Movement to Speech

What exactly happens from the moment a user performs a gesture until text or speech is produced?

1.  **Capture:** The user signs into the webcam. The frontend JavaScript captures the frame and sends it over a WebSocket.
2.  **Extraction:** The Python server receives the frame and passes it to **MediaPipe**, which locates the user's joints and extracts 166 precise 3D spatial coordinates.
3.  **Filtration:** The coordinates are checked by the **Anti-Stationary Variance Filter**. If the hands are moving, they are appended to a rolling 30-frame sequence buffer.
4.  **Inference:** The 30-frame sequence (representing ~1 second of motion) is fed into the **GRU Neural Network**, which outputs a predicted sign and a confidence percentage.
5.  **Translation:** If confidence is $>70\%$, the predicted English word is passed through the **Google NMT Engine** to translate it into the user's selected language (e.g., Spanish, French).
6.  **Delivery:** The server emits the final text back to the frontend. The UI instantly displays the translated caption, and the browser's native **Web Speech API (TTS)** synthesizes and speaks the word aloud.

---

## 10. Key Concepts: CV, OpenCV & MediaPipe

*   **Computer Vision (CV):** A field of artificial intelligence that enables computers to derive meaningful information from digital images, videos, and other visual inputs, and take actions based on that information.
*   **OpenCV:** An open-source computer vision library. In SignVision, it is utilized primarily in the desktop/training modules for raw image processing, frame capturing, and drawing visual skeletal overlays.
*   **MediaPipe:** A framework built by Google for building multimodal machine learning pipelines. We use its "Holistic" model, which rapidly identifies complex topological landmarks (face mesh, hands, body pose) from flat 2D images.

---

## 11. Environmental Robustness (Backgrounds & Lighting)

### How does the system handle different backgrounds?
**Extremely well.** Because MediaPipe is trained on massive, diverse datasets to detect human topological features, it effectively isolates the person from the background. By passing only the extracted 3D spatial coordinates (the "skeleton") to our GRU model—rather than the raw RGB video pixels—our temporal classification model becomes **100% agnostic to the background environment**. A cluttered room or a blank green screen yields the exact same coordinate array.

### How does lighting affect the system?
**Lighting is the primary bottleneck.** MediaPipe relies on visual contrast to identify hands and facial features. Extremely low lighting, harsh backlighting, or severe shadows can degrade landmark tracking accuracy. If MediaPipe cannot detect the hand, the coordinates drop to zero, corrupting the sequence fed to the GRU. However, we simulated sensor jitter during training (via Gaussian noise injection), making the GRU model highly robust to the slight tracking errors caused by moderate, real-world lighting variations.

---

## 12. Future Iterations & Improvements

If developed again from scratch, several architectural shifts would be implemented to scale the platform:

1.  **Client-Side Edge Processing:** Shift the MediaPipe extraction from the server to the browser (using MediaPipe WebAssembly). This would allow the client to send only lightweight float arrays over WebSockets instead of JPEG frames, reducing server bandwidth by 99% and enabling horizontal scaling to thousands of concurrent users.
2.  **Continuous Sign Language Recognition (CSLR):** Upgrade from isolated word classification to continuous sentence-level recognition by implementing Connectionist Temporal Classification (CTC) loss or an Attention-based Transformer architecture.
3.  **Dynamic Sequence Lengths:** Transition from a strict 30-frame window to dynamic time-distributed processing to handle signs of drastically varying speeds more naturally.
