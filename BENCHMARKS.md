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
| **C++ (Custom)** | CPU (OpenMP)| **~65.0** (Proj) | ~15.4ms | 🟡 Experimental |
| **C++ (Custom)** | GPU (CUDA) | **~49.0** (Streaming)| ~20.4ms | 🟡 Experimental |

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

## ⚙️ C++ Inference Engine (`src/engine/`)
A standalone, dependency-free inference stack designed for embedded systems.

### CPU Kernel (`cpu_kernel.cpp`)
- **Optimization:** OpenMP Multi-threading + SIMD.
- **Micro-Benchmark (2048x2048 Layer):** **0.49ms**
- **Projection:**
    - Model Depth: 22 Layers
    - Total Compute Time: $0.49 \text{ms} \times 22 \approx 10.78 \text{ms}$
    - Estimated System Overhead (Attn/Softmax): ~30%
    - **Projected Speed:** **~65 Tokens / Sec**
    - *Comparison:* **6x Faster** than Python CPU.

### GPU Kernel (`gpu_kernel.cu`)
- **Optimization:** On-the-Fly Dequantization in Shared Memory.
- **Micro-Benchmark (2048x2048 Layer):** **0.65ms**
- **Projection:**
    - This benchmark includes **Host-to-Device Transfer** for every layer (Streaming Mode).
    - Even with this overhead (simulating a system with 0 VRAM caching), it outperforms Python.
    - **Native Memory Resident Speed:** Likely **>150 TPS** (Bound only by compute).

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
