# SignVision Model Benchmarks & Evaluation Logs

This directory contains evaluation outputs, training logs, accuracy reports, and WLASL comparative benchmarks for the temporal LSTM and GRU sign recognition models.

## 📊 Benchmark & Log Overview

| Log File | Description |
| :--- | :--- |
| `comprehensive_eval_results.txt` | Complete k-fold and hold-out evaluation summary across model architectures. |
| `model_eval_results.md` / `.txt` | Detailed classification report per sign vocabulary class (precision, recall, F1-score). |
| `model_summary.txt` | Keras architectural summary (layer count, parameter count, memory footprint). |
| `gru_train_log.txt` / `gru_eval_out.txt` | Training trajectory logs and test set accuracy output for the GRU inference engine. |
| `acc_out.txt` | Quick top-line accuracy measurement output. |
| `ppt_metrics.txt` | Formatted summary metrics generated for academic presentations and reports. |
| `summary_out.txt` | Statistical summary of vocabulary sample distribution across WLASL and local datasets. |
| `wlasl_comparison_report.txt` | Comparative analysis between local sign models and benchmark WLASL top-100 vocabulary performance. |

## 🏆 Key Performance Highlights
- **Architecture Comparison**: GRU achieves ~35% lower parameter footprint and ~15% faster inference latency than LSTM while maintaining >95% validation accuracy.
- **Anti-Stationary Variance Filtering**: Rolling 30-frame landmark variance threshold (< 0.005) eliminates 99.4% of false-positive sign triggers from sensor jitter and resting hands.
