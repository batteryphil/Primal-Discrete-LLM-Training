#pragma once
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Magic Header
const char MAGIC[] = "TRIN";

// Unpacking LUT: 4-bit Nibble -> Prime Value
// Grid: {0, ±0.5, ±0.33, ±0.2, ±0.14, ...} -> 7 values + padding
// 0: 0.0
// 1: +0.5 (1/2)
// 2: -0.5
// 3: +0.333333 (1/3)
// 4: -0.333333
// 5: +0.2 (1/5)
// 6: -0.2
// 7-15: 0.0 (Reserved/Padding)
const float LUT[16] = {0.0f,  0.5f, -0.5f, 0.333333f, -0.333333f, 0.2f,
                       -0.2f, 0.0f, 0.0f,  0.0f,      0.0f,       0.0f,
                       0.0f,  0.0f, 0.0f,  0.0f};

struct Tensor {
  std::string name;
  int rows, cols;
  float scale = 1.0f;               // Scaling factor (Per-Tensor)
  std::vector<uint8_t> data_packed; // 4-bit packed data (Now 2 weights/byte)
  std::vector<float> data_fp32;     // Unpacked data (for CPU cache/debug)
};

class TrinityEngine {
public:
  std::vector<Tensor> weights;
  bool use_gpu = false;

  // Load the binary model file (trinity_1.58bit_packed.bin)
  void load(const std::string &path);

  // Forward pass: Input -> Output (Logits)
  // For now, this will just be a placeholder or simple matmul test
  void forward(const std::vector<float> &input, std::vector<float> &output);

  // CPU Kernel: Unpack and MatMul
  void cpu_forward(const std::vector<float> &input, std::vector<float> &output);

  // GPU Kernel: (To be implemented in .cu)
  void gpu_forward(const std::vector<float> &input, std::vector<float> &output);
};
