#pragma once
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


// Magic Header
const char MAGIC[] = "TRIN";

// Unpacking LUT: 0->0, 1->1, 2->-1, 3->0
// This maps the 2-bit packed values to their float representations
const float LUT[4] = {0.0f, 1.0f, -1.0f, 0.0f};

struct Tensor {
  std::string name;
  int rows, cols;
  std::vector<uint8_t> data_packed; // 2-bit packed data
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
