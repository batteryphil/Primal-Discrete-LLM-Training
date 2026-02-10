# Project Trinity: Phase 6 Roadmap

## 🎯 Target Alpha: The "Speed" Run (C++ Kernel)
**Objective:** Eliminate the Python overhead. Run the 246MB binary on bare metal.
- [ ] Design `trinity.cpp` loader (Header-only C++).
- [ ] Implement `MatMul_2bit` kernel (SIMD optimized).
- [ ] Port to ESP32 / Raspberry Pi Zero.

## 🎯 Target Beta: The "Big" Brain (Llama-3-8B)
**Objective:** Scale Prime Harmonic Evolution to 8B parameters.
- [ ] Test Prime Grid stability on Llama-3 layers.
- [ ] Scale SFT dataset to >1M tokens (prevent overfitting).
- [ ] Target Perplexity < 10.0.
