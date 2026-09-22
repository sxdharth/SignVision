# SignVision — Errors Detected & Solutions Applied

> **Purpose**: This file documents all bugs found and fixes applied across the project.
> Any future edits to these files **MUST** consult this document first to avoid re-introducing these issues.

---

## 1. Duplicate Event Handlers

| File | `web/webrtc/script.js` |
|------|------------------------|
| **Error** | Two separate `socket.on('prediction')` handlers existed. Both fired for every prediction event, causing double processing. The second handler used **undeclared variables** (`peakConf`, `peakText`, `peakSid`, `peakWindow`, `resetPeak`) → `ReferenceError` at runtime. |
| **Fix** | Removed the broken duplicate handler. Kept the score-accumulator algorithm (first handler). |
| **Rule** | Never register multiple `socket.on()` handlers for the same event. Always search the file for existing handlers before adding new ones. |

---

## 2. LSTM Buffer Ghosting (Phantom Predictions)

| File | `web/webrtc/script.js` |
|------|------------------------|
| **Error** | After a sign was committed, the backend LSTM buffer still contained old frames. On unlock, it immediately re-predicted the same (or wrong) sign from leftover data — even when the user was not signing. |
| **Fix** | (a) `socket.emit('clear_buffer')` after each commit clears the LSTM. (b) Added 2-second post-commit quiet period (`POST_COMMIT_QUIET_MS`) during which all predictions are ignored. (c) Raised noise gate from 35% → 50%. |
| **Rule** | After any sign is committed, **always** clear the backend buffer and enforce a quiet period before accepting new predictions. The noise gate should never be set below 50%. |

---

## 3. Frame Resolution Mismatch (Training vs. Inference)

| File | `web/webrtc/script.js` |
|------|------------------------|
| **Error** | Training used raw full-resolution frames from `cv2.VideoCapture(0)` (typically 640×480). Live inference was sending frames at **1/4 resolution with 50% JPEG quality**, degrading MediaPipe landmark detection and causing wrong predictions. |
| **Fix** | Changed `startTranslation()` to send frames at **640×480** with **80% JPEG quality**. |
| **Rule** | Frame resolution and quality in the live pipeline **must match** training conditions. Currently: `640×480, JPEG 80%`. If training resolution changes, update `startTranslation()` to match. |

---

## 4. Undefined CSS Variables

| File | `web/smart_home.html` |
|------|------------------------|
| **Error** | `toggleAutomation()` set `btn.style.background = 'var(--accent)'`, but `--accent` was **never defined** in the file's `<style>` block. Button background became invisible. |
| **Fix** | Replaced with explicit color `rgba(16, 185, 129, 0.15)` matching the green theme. |
| **Rule** | Never use CSS custom properties (`var(--x)`) in inline JavaScript styles unless the variable is guaranteed to be defined in the same document's stylesheet. |

---

## 5. Dead Code — No-Op Socket Emit

| File | `web/smart_home.html` |
|------|------------------------|
| **Error** | `socket.emit('prediction', { text: offCmd, conf: 1.0 })` in `toggleAutomation()` — the server has **no handler** for client-originated `prediction` events. This line did nothing. |
| **Fix** | Removed the dead code. Device state and UI are updated directly. |
| **Rule** | Before using `socket.emit('eventName')`, verify that the server has a corresponding `@sio.on('eventName')` handler. |

---

## 6. Duplicate Imports

| File | `src/inference_engine.py` |
|------|---------------------------|
| **Error** | `import os` appeared **three times** (lines 3, 9, 10). |
| **Fix** | Removed the two duplicates. |
| **Rule** | Keep imports at the top of the file and never duplicate them. |

---

## 7. Missing CSS Standard Properties

| Files | `web/webrtc/home.html`, `web/webrtc/landing.html` |
|-------|-----------------------------------------------------|
| **Error** | Used `-webkit-background-clip: text` without the standard `background-clip: text`. Non-WebKit browsers would not render gradient text correctly. |
| **Fix** | Added `background-clip: text` alongside the vendor prefix. |
| **Rule** | Always include the standard CSS property alongside any `-webkit-` prefixed version. |

---

## 8. Translation Span Not Reset

| File | `web/webrtc/script.js` |
|------|------------------------|
| **Error** | `toggleTranslation()` did not hide or clear `#masterCaptionTranslation` when disabling translation. Old translated text persisted on screen. |
| **Fix** | Added cleanup: `transEl.style.display = 'none'; transEl.innerText = '';` |
| **Rule** | When toggling any UI feature OFF, always reset **all** related DOM elements to their default state. |

---

## 9. Inconsistent Language Options

| Files | `web/webrtc/index.html` vs `web/webrtc/landing.html` |
|-------|--------------------------------------------------------|
| **Error** | Pre-call modal offered English, Hindi, Malayalam. Sidebar `#langSelect` dropdown was missing Malayalam. |
| **Fix** | Added `<option value="ml">Malayalam</option>` to the sidebar dropdown. |
| **Rule** | Language options must be consistent across all UI surfaces. When adding a language to one location, add it everywhere. |

---

## 10. Emoji Encoding Corruption

| File | `web/webrtc/landing.html` |
|------|---------------------------|
| **Error** | Emoji flags (🇬🇧, 🇮🇳, 🌴) and `→` symbol displayed as garbled characters on some Windows setups due to file encoding inconsistencies. |
| **Fix** | Replaced emojis with ASCII text (`EN`, `HI`, `ML`) and `→` with `&rarr;` HTML entity. |
| **Rule** | Avoid raw Unicode emoji in HTML files. Use HTML entities or icon fonts (Ionicons) instead for maximum compatibility. |

---

## Quick Reference: Critical Pipeline Parameters

| Parameter | Value | File | Notes |
|-----------|-------|------|-------|
| Frame resolution | 640×480 | `script.js` | Must match training |
| JPEG quality | 80% | `script.js` | Must match training |
| Frame rate | 30 FPS (33ms) | `script.js` | Must match training |
| Noise gate | 50% | `script.js` | Per-prediction minimum |
| Score accumulator window | 1000ms | `script.js` | Collection period |
| Same-sign cooldown | 3000ms | `script.js` | Prevents repeat fires |
| Post-commit quiet period | 2000ms | `script.js` | Buffer refresh time |
| Anti-ghost check frames | 5 | `inference_engine.py` | Last N frames checked for zero-hands |
| LSTM sequence length | 30 | `inference_engine.py` | `MAX_LENGTH` |
| Inference throttle | 0.3s | `inference_engine.py` | Min gap between `model.predict()` calls |

---

## 11. Model Class Confusion (Training Data Source Mismatch)

| Files | `tools/video_call_data_merger.py`, `Data/Custom_Processed/`, Model |
|-------|-------------------------------------------------------------------|
| **Error** | User signed `thanks` but model predicted `how` at 100% confidence. The training data (100 samples per class) was sourced from `Data/Custom_Processed/` (likely WLASL external data), NOT from the user's own webcam via `video_call_data_recorder.py`. External sign datasets contain different people's signing styles, which don't match the user's specific hand shapes and movements. |
| **Fix** | **Requires retraining**: user must record their own samples using `video_call_data_recorder.py` (which saves to `Data/Video_Call_Raw/`), then run `retrain_video_call.py` (which merges from `Video_Call_Raw/` and retrains). This bypasses the `Custom_Processed` data. |
| **Rule** | For personalized accuracy, always train on the **user's own webcam recordings** via `video_call_data_recorder.py` + `retrain_video_call.py`. Do NOT rely on external datasets (`Custom_Processed/`) for production use — they produce class confusion. |

---

## 12. Model Overfitting — 100% Test Accuracy but Fails Live

| Files | `tools/retrain_video_call.py`, `tools/video_call_trainer.py` |
|-------|--------------------------------------------------------------|
| **Error** | Model achieved 100% accuracy on test data (700 samples, 7 classes) but misclassified signs in the live video call. This is **classic overfitting**: the model memorized training data but couldn't generalize to the slight differences caused by JPEG compression noise, MediaPipe landmark variance, and temporal frame jitter in the live pipeline. |
| **Fix** | Added **5× data augmentation** to `retrain_video_call.py`: (1) Gaussian noise σ=0.02 simulating JPEG + MediaPipe variance, (2) Stronger noise σ=0.04, (3) Temporal jitter (shift frames by 1-2 positions), (4) Random hand landmark dropout. Increased dropout 0.4→0.5 and L2 regularization 0.005→0.01. **User must retrain** with `retrain_video_call.py` for fix to take effect. |
| **Rule** | 100% test accuracy on small datasets (&lt;1000 samples) is a red flag for overfitting. Always use data augmentation. Target 95-98% test accuracy — not 100%. |

---

## 13. Constant "Hello" False Positives on Idle Webcam Frames

| Files | `src/inference_engine.py`, `web/webrtc/script.js` |
|-------|---------------------------------------------------|
| **Error** | The generalized `video_call_model.h5` constantly predicted "hello" when the user was sitting still or making no obvious signs. The training data for the general model does not contain a "none" or "idle" class, and sensor noise/autofocus jitter bypassed basic zero-motion strict checks. |
| **Fix** | Implemented an `Anti-Stationary Filter` in `src/inference_engine.py` that calculates the mathematical variance of landmark movement across the rolling 30-frame sequence. Tuned the variance threshold to `< 0.005` to ignore webcam jitter, and increased the frontend confidence threshold from `0.50` to `0.70`. |
| **Rule** | Always filter low-variance stationary sequences before feeding frames into temporal RNN/LSTM sequence classifiers. |

---

## 14. WebSocket Server Disconnections / "aiohttp" Crash Under Load

| Files | `web/webrtc/script.js`, `web/webrtc/server.py` |
|-------|------------------------------------------------|
| **Error** | Using `setInterval` to fire 30 FPS to the backend caused a massive queue of pending bytes to pile up in memory because LSTM/GRU inference took ~50-100ms per frame under heavy load, causing `aiohttp` to crash with an `OSError`. |
| **Fix** | Switched `startTranslation` camera loop in `script.js` to an asynchronous `requestAnimationFrame` wait pattern. |
| **Rule** | Never use unthrottled timer loops (`setInterval`) for real-time video frame streaming over WebSockets. Always use async frame gating. |

---

## 15. Frontend Permanent "Listening..." State Freeze

| File | `web/webrtc/script.js` |
|------|------------------------|
| **Error** | The UI permanently froze on "Listening..." because a dropped WebSocket packet left an `isProcessingFrame` boolean flag permanently locked. |
| **Fix** | Removed the brittle `isProcessingFrame` tracking boolean entirely and implemented a mathematically stable `await new Promise(r => setTimeout(r, 66))` delay (~15-20 FPS) without requiring server ack unlocking. |
| **Rule** | Avoid client-side lock booleans that depend on network packet return confirmation for unlock. Use time-bounded asynchronous delays. |

