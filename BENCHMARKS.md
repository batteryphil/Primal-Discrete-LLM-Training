# 📊 Project Trinity: Performance Benchmarks (V1.0.3)

**Hardware:**
- **CPU:** Standard x86_64 (AVX2 Support)
- **GPU:** NVIDIA GeForce GTX 1080 Ti (11GB)
- **Model:** Trinity-1.1B (1.58-bit / 2.2GB $\rightarrow$ 246MB)

## 🏆 Summary
| Engine | Platform | Speed (TPS) | Latency / Token | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Python (PyTorch)** | GPU (CUDA) | **35.08** | 28.5ms | 🟢 Production |
| **Python (PyTorch)** | CPU (Int8) | **11.05** | 90.5ms | 🟢 Playable |
| **C++ (4-bit Prime)**| CPU (OpenMP)| **~61.0** (Proj) | ~16.4ms | 🟡 V2.0.0 (Valid) |
| **C++ (4-bit Prime)**| GPU (CUDA) | **~35.0** (Streaming)| ~32.0ms | 🟡 V2.0.0 (Valid) |

---

## 🐍 Python Engine (`src/run_inference.py`)
The reference implementation uses PyTorch. V1.0.3 introduces specific optimizations for both backends.

### GPU Optimization (Freeze-Quant)
- **Technique:** Weights are frozen to `{ -1, 0, +1 }` during the first pass, skipping the expensive quantization search for all subsequent tokens.
- **Throughput:** **35.08 tokens/sec**. Matches native FP16 execution speed for this model size.

### CPU Optimization (Pre-Freeze + Int8)
- **Technique:**
    1.  **Pre-Freeze:** Initialization skips the O(N) grid search, reducing startup from ~120s to **<1s**.
    2.  **Dynamic Quantization:** Layers are converted to Int8 to leverage **AVX2/AVX512** instructions.
- **Throughput:** **11.05 tokens/sec**. Up from ~1.5 TPS in the prototype phase.

---

## ⚙️ C++ Inference Engine (V2.0.0)
A standalone, dependency-free inference stack updated to support **Valid 4-bit Prime Quantization**.

### CPU Kernel (`cpu_kernel.cpp`)
- **Optimization:** OpenMP + SIMD. Unpacks 4-bit nibbles (2 weights/byte) and applies per-tensor scaling.
- **Micro-Benchmark (2048x2048 Layer):** **0.50ms**
- **Projection:**
    - Model Depth: 22 Layers
    - Total Compute Time: $0.50 \text{ms} \times 22 \approx 11.0 \text{ms}$
    - Estimated Overhead: ~30%
    - **Projected Speed:** **~61 Tokens / Sec**

### GPU Kernel (`gpu_kernel.cu`)
- **Optimization:** On-the-Fly 4-bit Dequantization in Shared Memory.
- **Micro-Benchmark (2048x2048 Layer):** **1.02ms**
- **Analysis:**
    - Latency increased vs V1.0.3 (0.65ms) because memory bandwidth usage doubled (2-bit -> 4-bit).
    - However, this kernel is **Mathematically Correct**, supporting the full 7-value Prime Grid, whereas V1.0.3 was a limited ternary implementation.
    - **Projected Speed:** **~35-40 TPS** (Streaming Mode).

## 📉 Reproducibility
To verify these numbers on your own hardware:

**Python:**
```bash
python src/run_inference.py --model models/trinity_1.58bit_packed.bin
```

**C++:**
```cmd
cd src\engine
run_test.bat
```
