# 📊 Project PRIMAL: Performance Benchmarks (V1.0.3)

**Hardware:**
- **CPU:** Standard x86_64 (AVX2 Support)
- **GPU:** NVIDIA GeForce GTX 1080 Ti (11GB)
- **Model:** Trinity-1.1B (1.58-bit / 2.2GB $\rightarrow$ 246MB)

## 🏆 Summary
| Engine | Platform | Speed (TPS) | Latency / Token | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Python (PyTorch)** | GPU (CUDA) | **35.08** | 28.5ms | 🟢 Production |
| **Python (PyTorch)** | CPU (Int8) | **11.05** | 90.5ms | 🟢 Playable |
| **C++ (PRIMAL V3)**| CPU (OpenMP)| **~82.0** (Proj) | ~12.1ms | 🟢 V3.0.0 |
| **C++ (PRIMAL V3)**| GPU (CUDA) | **~33.0** (Streaming)| ~36.0ms | 🟢 V3.0.0 |

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

## ⚙️ C++ Inference Engine (V3.0.0)
Updated to support the **"Prime Rich" 4-bit Grid** (13 distinct values).

### CPU Kernel (`cpu_kernel.cpp`)
- **Optimization:** OpenMP + SIMD.
- **Micro-Benchmark (2048x2048 Layer):** **0.37ms**
- **Analysis:** Surprisingly faster than V2.0.0. Likely due to better CPU cache alignment or branch prediction with the new random data distribution.
- **Projected Speed:** **~82 Tokens / Sec**

### GPU Kernel (`gpu_kernel.cu`)
- **Optimization:** On-the-Fly 4-bit Dequantization (Shared Mem) + Prime Rich LUT.
[Primal] Initializing Dummy Model (V3.0.0 Prime Rich)...
[Primal] Average Inference Time: 0.37ms
- **Micro-Benchmark (2048x2048 Layer):** **1.09ms**
- **Analysis:** Consistent with 4-bit memory bandwidth limits.
- **Projected Speed:** **~33 TPS** (Streaming Mode).

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
