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
// Grid: V3.0.0 "Prime Rich"
const float LUT[16] = {
    0.0f,                  // 0x0: Sparsity
    1.0f,      -1.0f,      // 0x1, 0x2: Unity
    0.5f,      -0.5f,      // 0x3, 0x4: 1/2
    0.333333f, -0.333333f, // 0x5, 0x6: 1/3
    0.2f,      -0.2f,      // 0x7, 0x8: 1/5
    0.142857f, -0.142857f, // 0x9, 0xA: 1/7
    0.090909f, -0.090909f, // 0xB, 0xC: 1/11
    0.076923f, -0.076923f, // 0xD, 0xE: 1/13
    0.0f                   // 0xF: Reserved
};

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
