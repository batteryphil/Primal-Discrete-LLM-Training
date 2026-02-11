#pragma once
#include <string>
#include <vector>


// V3.0.0 "Prime Rich" LUT: 4-bit Nibble -> Prime Value
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
  float scale = 1.0f;
  std::vector<uint8_t> data_packed;
  std::vector<float> data_fp32;
};

class PrimalEngine {
public:
  std::vector<Tensor> weights;
  bool use_gpu = false;
  void load(const std::string &path);
  void forward(const std::vector<float> &input, std::vector<float> &output);
  void cpu_forward(const std::vector<float> &input, std::vector<float> &output);
  void gpu_forward(const std::vector<float> &input, std::vector<float> &output);
};
