# Contributing to SignVision

We love open-source contributions! Whether you want to add new ASL sign vocabulary, optimize temporal neural network architectures, or enhance the WebRTC UI, your contributions help make communication accessible for everyone.

---

## 🛠️ Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/<your-username>/SignVision.git
   cd SignVision
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify Your Setup**:
   Run the CLI help command to confirm dependencies are loaded:
   ```bash
   python signvision.py --help
   ```

---

## 🤟 Adding New ASL Vocabulary & Training Custom Models

To train the model on your own hand gestures or add new words:

1. **Record Training Samples**:
   Use our interactive recorder script to capture 30-frame temporal sequences for your vocabulary words:
   ```bash
   python tools/video_call_data_recorder.py
   ```
   Samples are automatically saved to `Data/Video_Call_Raw/`.

2. **Retrain the Sequence Classifier**:
   Run the retrain script with 5× data augmentation (Gaussian coordinate noise, temporal jitter, and landmark dropout):
   ```bash
   python tools/retrain_video_call.py
   ```
   This will train the GRU/LSTM model and update `Models/video_call_model_gru.h5`.

3. **Verify Evaluation Metrics**:
   Check validation accuracy and per-class F1-scores:
   ```bash
   python tools/diagnostics/check_model_accuracy.py
   ```

---

## 🐞 Bug Reports & Pull Requests

* **Before submitting a bug report**: Check `docs/ERRORS_AND_FIXES.md` to see if the issue is a known behavior or regression guardrail.
* **Pull Request Guidelines**:
  - Keep code clean and well-commented.
  - Follow existing architectural conventions (e.g., MediaPipe 166-feature vectors, rolling 30-frame window).
  - Document any pipeline parameter changes in `docs/ERRORS_AND_FIXES.md`.
