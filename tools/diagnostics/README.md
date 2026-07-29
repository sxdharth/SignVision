# SignVision Diagnostic & Sanity Check Tools

This directory contains standalone utility scripts for verifying datasets, inspecting model weights, testing hardware relay connectivity, and diagnosing real-time inference latency.

## 🧰 Available Diagnostic Scripts

| Script | Description |
| :--- | :--- |
| `check_data.py` | Inspects shape, class distributions, and sample counts in raw and preprocessed `.npy` dataset arrays. |
| `check_data_bias.py` | Evaluates class balance and identifies potential sampling bias across WLASL and custom vocabulary sets. |
| `check_gru.py` / `check_gru2.py` | Verifies GRU model weight loading, input/output tensor shapes, and sequence padding consistency. |
| `check_model_accuracy.py` | Evaluates trained LSTM/GRU `.h5` models against validation split datasets and reports accuracy metrics. |
| `diagnose_inference.py` | Benchmarks live MediaPipe landmark extraction speed and temporal sequence inference latency. |
| `diagnose_pipeline.py` / `pipeline_diagnosis.py` | Performs end-to-end sanity checking across the camera capture, feature extraction, and prediction stages. |
| `eval_output.py` / `get_metrics_for_ppt.py` | Generates formatted classification reports, confusion matrices, and metrics summaries for presentations. |
| `test_feature_extractor.py` | Validates MediaPipe holistic 166-feature vector generation on test frames. |
| `test_relay.py` | Tests serial communication with Arduino Uno/Nano relay boards for Hardware-in-the-Loop (HITL) smart home automation. |

## 🚀 Usage Example

To check model accuracy on your local dataset:
```bash
python tools/diagnostics/check_model_accuracy.py
```

To test serial connectivity with the Arduino relay module:
```bash
python tools/diagnostics/test_relay.py
```
