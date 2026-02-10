# ⟁ Trinity: The "Homebrew" 1.58-bit LLM

**Trinity-1.1B** is a proof-of-concept 1.58-bit model evolved from TinyLlama using **Prime Harmonic Evolution**. 
Unlike standard BitNet models trained from scratch, Trinity was "snapped" to a Prime Grid `{±1/p, 0}` from a pre-trained FP16 checkpoint in under 500 steps.

## ⚡ The Specs
| Feature | Original (TinyLlama) | **Trinity (1.58-bit)** |
| :--- | :--- | :--- |
| **Size** | 2.2 GB | **246 MB** |
| **Compression** | 1x | **8.51x** |
| **Perplexity** | ~8.0 | **15.3** (WikiText) |
| **Training Device** | GPU Cluster | **Single Consumer GPU** |

## 🧪 How It Works
We used a custom **Gradient Evolution** technique to migrate weights to a rigid Prime Reciprocal Grid. This prevents the "brain death" usually seen in extreme Post-Training Quantization (PTQ). 
The model was then Instruction Tuned on Alpaca to recover factual associations (e.g., "Paris is the capital of France").

## 🚀 Usage
This model is packed into a custom 2-bit binary format.
To run it, you need the unpacking script included in this repo.

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Unpack and Run (CPU Inference)
python src/run_inference.py --model models/trinity_1.58bit_packed.bin
```

## 📄 Documentation
For deep technical details, mathematical proofs of the Prime Grid, and training methodologies, see:
- [PAPER.md](./PAPER.md) - Formal Scientific Draft.

## 📜 License
- Code: Apache 2.0
- Weights: CC BY-NC 4.0

## Acknowledgements
Developed by **BatteryPhil** using the Project Trinity Evolutionary Protocol.
Special thanks to the open-source community for TinyLlama and BitNet research benchmarks.
