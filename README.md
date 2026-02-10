# Trinity-1.58bit: Prime Harmonic LLM Evolution

Trinity-1.58bit is a research framework for evolving standard Large Language Models into high-density **Prime Harmonic Grids**. By aligning weights to prime reciprocals, we achieve significant compression without the need for pre-training from scratch.

## 🚀 Key Results
- **Model:** TinyLlama-1.1B
- **Architecture:** 1.58-bit (Ternary) Prime-Aligned
- **Convergence Loss:** 2.73
- **Compression Ratio:** **8.51x** (2.2GB -> 246MB)
- **Target Hardware:** Edge devices, Raspberry Pi, Mobile.

## 🛠️ Repository Structure
- `src/train.py`: The Instruction Tuning / Evolution engine.
- `src/pack.py`: Binary packing protocol for 2-bit storage.
- `models/`: Contains the packed `.bin` artifact.
- `PAPER.md`: Detailed scientific methodology.

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🧪 Quick Start (Inference)
The model uses a custom `PrimeLinear` layer to map weights back to the Prime Grid at runtime. See `PAPER.md` for technical depth.

## 📜 License
This project is licensed under a hybrid Apache 2.0 / CC-BY-NC 4.0 license. See `LICENSE` for details.
