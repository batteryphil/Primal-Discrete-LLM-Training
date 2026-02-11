#include "primal.h"
#include <cstdlib>
#include <iostream>
#include <omp.h>


void PrimalEngine::load(const std::string &path) {
  std::cout << "[Primal] Initializing Dummy Model (V3.0.0 Prime Rich)..."
            << std::endl;
  Tensor t;
  t.rows = 2048;
  t.cols = 2048;
  t.scale = 0.002f;
  size_t num_weights = t.rows * t.cols;
  size_t packed_size = num_weights / 2; // 4-bit = 2 weights per byte
  t.data_packed.resize(packed_size);

  // Fill with random data (0x00 - 0xFF) to test full LUT range
  for (size_t i = 0; i < packed_size; ++i) {
    t.data_packed[i] = rand() % 256;
  }

  t.data_fp32.resize(num_weights);
// Unpack for CPU validation
#pragma omp parallel for
  for (long long i = 0; i < (long long)packed_size; ++i) {
    uint8_t byte = t.data_packed[i];
    size_t idx = i * 2;
    t.data_fp32[idx + 0] = LUT[(byte >> 4) & 0x0F]; // High Nibble
    t.data_fp32[idx + 1] = LUT[(byte >> 0) & 0x0F]; // Low Nibble
  }
  weights.push_back(t);
}

void PrimalEngine::forward(const std::vector<float> &input,
                           std::vector<float> &output) {
  if (use_gpu)
    gpu_forward(input, output);
  else
    cpu_forward(input, output);
}

void PrimalEngine::cpu_forward(const std::vector<float> &input,
                               std::vector<float> &output) {
  if (weights.empty())
    return;
  const Tensor &W = weights[0];
  output.resize(W.rows);
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < W.rows; ++i) {
    float sum = 0.0f;
    for (int k = 0; k < W.cols; ++k) {
      sum += W.data_fp32[i * W.cols + k] * input[k];
    }
    output[i] = sum * W.scale;
  }
}
