#pragma once
#include <cstdint> // <--- CRITICAL FIX: Required for uint8_t
#include <string>
#include <vector>

// V3.0.0 "Prime Rich" LUT: 4-bit Nibble -> Prime Harmonic Value
const float LUT[16] = {
    0.0f,                  // 0x0: Sparsity (Zero)
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

  // GPU Resident Data (Phase 20)
  uint8_t *d_data = nullptr;
};

class PrimalEngine {
public:
  std::vector<Tensor> weights;
  bool use_gpu = false;

  // GPU Workspace (Ping-Pong)
  float *d_ping = nullptr;
  float *d_pong = nullptr;
  size_t workspace_size = 0;

  void init_workspace(size_t size);
  void free_workspace();

  void load(const std::string &path);
  void forward(const std::vector<float> &input, std::vector<float> &output);

  // Backends
  void cpu_forward(const std::vector<float> &input, std::vector<float> &output);
  void gpu_forward(const std::vector<float> &input, std::vector<float> &output);

  // Phase 21: Zero-Copy Streaming
  // We use void* for stream to avoid including cuda_runtime.h in header if
  // possible, but we already use uint8_t, etc. Let's strictly use void* for the
  // stream member here or forward declare. Actually, we can just use void*
  // stream = nullptr and cast it in .cu file to simplify dependencies.
  void *stream = nullptr;
  void gpu_forward_streamed(const float *h_input, float *h_output,
                            int input_size);
};
