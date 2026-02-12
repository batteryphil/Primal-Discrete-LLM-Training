# ⟁ Primal-Discrete-LLM-Training

**PRIMAL-1.1B** is a proof-of-concept **4-bit Prime Harmonic** model evolved from TinyLlama.

> **[V3.0.0 Prime Rich Update]**
> Building on the valid 4-bit architecture of V2.0.0, **V3.0.0** expands the quantization grid to include **13 distinct Prime Harmonic values** ($\pm 1, \pm 1/2, \pm 1/3, \pm 1/5, \pm 1/7, \pm 1/11, \pm 1/13$). This utilizes the previously empty slots in the 4-bit nibble to maximize precision without increasing file size.

## 🏆 Performance Profile (V3.0.0)
| Device | Engine | Speed (TPS) | VRAM / RAM | Status |
| :--- | :--- | :--- | :--- | :--- |
| **GPU (GTX 1080 Ti)** | Python (Native) | **35.08 TPS** | 850 MB | 🟢 Production |
| **CPU (i7-8700K)** | C++ (PRIMAL V3) | **~82.0 TPS*** | 580 MB | 🟢 Validated |
| **GPU (GTX 1080 Ti)** | C++ (PRIMAL V3) | **~33.0 TPS*** | 580 MB | 🟢 Validated |

## ⚡ The Specs (V3.0.0)
| Feature | Original (TinyLlama) | **PRIMAL (V3 Prime)** |
| :--- | :--- | :--- |
| **Size** | 2.2 GB | **550 MB** |
| **Compression** | 1x | **4.0x** |
| **Perplexity** | ~8.0 | **<15.3** (Proj) |
| **Grid** | FP16 | **13-Value Prime** |
| **C++ Engine** | N/A | **0.37ms / Layer** |

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
