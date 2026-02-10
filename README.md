# ⟁ Trinity: The "Homebrew" 1.58-bit LLM

**Trinity-1.1B** is a proof-of-concept 1.58-bit model evolved from TinyLlama using **Prime Harmonic Evolution**. 
Unlike standard BitNet models trained from scratch, Trinity was "snapped" to a Prime Grid `{±1/p, 0}` from a pre-trained FP16 checkpoint in under 500 steps.

## ⚡ The Specs (V1.0.3)
| Feature | Original (TinyLlama) | **Trinity (1.58-bit)** |
| :--- | :--- | :--- |
| **Size** | 2.2 GB | **246 MB** |
| **Compression** | 1x | **8.51x** |
| **Perplexity** | ~8.0 | **15.3** (WikiText) |
| **Inference (GPU)** | 35 TPS | **35.08 TPS** (Native) |
| **Inference (CPU)** | 5 TPS | **11.05 TPS** (Int8) |
| **C++ Engine** | N/A | **0.29ms / Layer** |

## 🧪 How It Works
We used a custom **Gradient Evolution** technique to migrate weights to a rigid Prime Reciprocal Grid. This prevents the "brain death" usually seen in extreme Post-Training Quantization (PTQ). 
The model was then Instruction Tuned on Alpaca to recover factual associations (e.g., "Paris is the capital of France").

## 🚀 Usage

### 1. Python Inference (Universal)
Runs on any system with PyTorch (CPU or GPU auto-detected).
```bash
# Install Dependencies
pip install -r requirements.txt

# Run Inference
python src/run_inference.py --model models/trinity_1.58bit_packed.bin
```

### 2. C++ Inference Engine (High-Performance)
Standalone engine with **0.29ms/layer** latency. Requires Visual Studio (Windows) or GCC (Linux).
```cmd
# Build and Run on Windows (VS Dev Prompt)
cd src\engine
build.bat
run_test.bat
```

## 📄 Documentation
For deep technical details, mathematical proofs of the Prime Grid, and training methodologies, see:
- [PAPER.md](./PAPER.md) - **Formal Scientific Draft (Updated for V1.0.3)**.
- [BENCHMARKS.md](./BENCHMARKS.md) - **Detailed CPU/GPU Performance Analysis**.

## 📜 License
- Code: Apache 2.0
- Weights: CC BY-NC 4.0

## Acknowledgements
Developed by **BatteryPhil** using the Project Trinity Evolutionary Protocol.
Special thanks to the open-source community for TinyLlama and BitNet research benchmarks.
