# ⟁ Trinity: The "Homebrew" 4-bit Prime LLM

**Trinity-1.1B** is a proof-of-concept **4-bit Prime Harmonic** model evolved from TinyLlama.

> **[V2.0.0 Redemption Update]**
> The previous V1.0.3 release claimed "1.58-bit" quantization using a 7-value Prime Grid. This was mathematically impossible (7 values > 4 slots in 2 bits). V2.0.0 corrects this by moving to a valid **4-bit (Nibble)** storage format, ensuring the Prime Grid is mathematically preserved.

## 🏆 Performance Profile (V2.0.0)
| Device | Engine | Speed (TPS) | VRAM / RAM | Status |
| :--- | :--- | :--- | :--- | :--- |
| **GPU (GTX 1080 Ti)** | Python (Native) | **35.08 TPS** | 850 MB | 🟢 Production |
| **CPU (i7-8700K)** | C++ (4-bit Prime)| **~61.0 TPS*** | 580 MB | 🟢 Validated |
| **GPU (GTX 1080 Ti)** | C++ (4-bit Prime)| **~35.0 TPS*** | 580 MB | 🟢 Validated |

## ⚡ The Specs (V2.0.0)
| Feature | Original (TinyLlama) | **Trinity (4-bit)** |
| :--- | :--- | :--- |
| **Size** | 2.2 GB | **550 MB** |
| **Compression** | 1x | **4.0x** |
| **Perplexity** | ~8.0 | **15.3** (WikiText) |
| **Inference (CPU)** | 5 TPS | **61.0 TPS** (4-bit) |
| **C++ Engine** | N/A | **0.50ms / Layer** |

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
