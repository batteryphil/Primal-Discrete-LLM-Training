# Project PRIMAL: 4-bit Prime-Harmonic Training Engine

> **Status:** Active Research / Proof of Concept
> **Hardware Target:** Consumer GPUs (e.g., GTX 1080 Ti, RTX 3060)
> **License:** MIT

## 🚀 The 11GB Challenge
Training Large Language Models (LLMs) usually requires massive VRAM because of the **Shadow Weight Tax- **Neural Vocoder:** HiFi-GAN v3 integration via SpeechBrain.
- **Automation:** `voice_test.py` for batch inference verification.

## 🚀 Quick Start (Inference)

To generate speech from text, ensure you are in the project root and run:

```powershell
# Single Sentence
python tts_inference.py --text "Project Trinity is alive." --checkpoint "checkpoints/ghost_tts/best_sentinel.pt" --output "output.wav"

# Automated Batch Test
python voice_test.py
```

Results will be saved in `tests/voice_samples/`.
ing (QAT) keeps the model in 4-bit but maintains a full FP16/FP32 copy of the weights for updates, effectively doubling memory usage.

**Project PRIMAL** removes the shadow weights entirely.

It implements a **Discrete Optimization Loop** that trains a 0.1B parameter model directly on a rigid 4-bit integer grid. This allows for massive batch sizes and high throughput on older cards like the GTX 1080 Ti.

---

## ⚡ Key Features

### 1. Prime Harmonic Grid (v3.0.0)
Instead of linear INT4 quantization (which wastes precision on large numbers), PRIMAL uses a custom 13-value Look-Up Table (LUT) derived from prime reciprocals. This concentrates precision around zero, where 90% of LLM weights reside.

| Index | Value | Description |
| :--- | :--- | :--- |
| **0-6** | `±1, ±0.5, ±0.33...` | Coarse adjustment for "body" layers. |
| **7** | `0.0` | Exact zero for sparsity. |
| **Fine** | `±0.66, ±0.25...` | (Layer 12 Only) High-precision bridge. |

### 2. The "Poltergeist" Method (Decoupled Flipping)
Discrete training often fails due to "stochastic thrashing" during gradient accumulation. PRIMAL solves this with **Decoupled Flipping**:
* **Backward Pass:** No updates. Gradients cast "votes" (`+1` or `-1`) into an `int8` buffer.
* **Optimizer Step:** Votes are aggregated. Weights only flip if there is a **consensus** across micro-batches (e.g., Batch 64).
* **Adaptive Probability:** Weights flip stochastically based on their magnitude (Z-Score filtering).

### 3. Efficiency Benchmarks (GTX 1080 Ti)
| Metric | Model Size | Result | Notes |
| :--- | :--- | :--- | :--- |
| **Training VRAM** | 0.1B Params | **10.3 GB** | @ Batch Size 64 (Full Saturation) |
| **Inference VRAM** | 1.1B Params | **~550 MB** | 4-bit Loading Verified |
| **Throughput** | 0.1B Params | **~6,000 TPS** | Training (Python Loop) |

---

## 🛠️ Installation & Usage

### Prerequisites
* NVIDIA GPU (Pascal or newer)
* CUDA 11.8+
* Python 3.10+

### Quick Start
```bash
# Clone the repo
git clone https://github.com/batteryphil/Primal-Discrete-LLM-Training.git
cd Primal-Discrete-LLM-Training

# Install dependencies
pip install -r requirements.txt

# Run the training demo (0.1B Model)
python primal_train_ghost.py
```
