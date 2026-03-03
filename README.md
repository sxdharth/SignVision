# SignVision: AI-Powered Accessible Communication Platform

*   **Real-Time Sign Language AI**: Instantly translates American Sign Language (ASL) and finger-spelling into text using advanced neural networks, bridging the gap between signers and non-signers.
*   **Inclusive WebRTC Video Calling**: A browser-based video platform featuring live gesture-to-text captions, ensuring seamless communication for Deaf, Hard of Hearing, and Speech-Impaired users.
*   **Adaptive Accessibility**: Offers personalized profiles (e.g., Blind, Deaf) and massive dataset training (WLASL) to deliver a robust, user-centric experience for all abilities.

## 🛠️ Technology Stack
*   **Core**: Python 3.11+
*   **AI/ML**: TensorFlow, Keras, MediaPipe, OpenCV.
*   **Web**: AIOHTTP, Python-SocketIO, HTML5, CSS3 (Glassmorphism Design).
*   **Real-Time Communication**: WebRTC (Peer-to-Peer Video), Socket.IO (Signaling).

## 📂 Project Structure
*   `src/`: Core Python application logic (`main_app.py`, `inference_engine.py`).
*   `web/`: WebRTC server and frontend files (`server.py`, `index.html`, `landing.html`).
*   `tools/`: Utilities for dataset processing and training (`wlasl_processor.py`, `model_trainer.py`).
*   `Data/` & `Models/`: Stores processed datasets and trained `.h5` models.

## 🚦 Getting Started

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Desktop App (Detection Only)
For local testing of the sign language detection system:
```bash
python run.py
```
*   **Controls**: `S` (Spelling Mode), `T` (TTS), `L` (Translate), `Q` (Quit).

### 3. Run the Video Call Platform
To start the web server:
```bash
python web/webrtc/server.py
```
1.  Open your browser to `http://localhost:8080`.
2.  Select your visibility profile (e.g., "Deaf / Hard of Hearing").
3.  Click **Join Call**.
4.  **Enable Captions**: Click the **Globe Icon** to see real-time translation of your partner's signs!
