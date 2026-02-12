#include "primal.h"
#include <iostream>
#include <omp.h>
#include <vector>


// PrimalEngine::load is in primal.cpp

void PrimalEngine::forward(const std::vector<float> &input,
                           std::vector<float> &output) {
  if (use_gpu) {
    gpu_forward(input, output);
    return;
  }
  cpu_forward(input, output);
}

void PrimalEngine::cpu_forward(const std::vector<float> &input,
                               std::vector<float> &output) {
  if (weights.empty())
    return;

  // 'x' is our Residual Stream. We start with the input.
  std::vector<float> x = input;
  std::vector<float> next_x;

  // Loop through every layer in the .primal file
  for (size_t l = 0; l < weights.size(); ++l) {
    const Tensor &W = weights[l];
    next_x.assign(W.rows, 0.0f);

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < W.rows; ++i) {
      float sum = 0.0f;
      // Matrix Multiplication
      // Note: This assumes W.cols matches x.size(). No check for perf.
      for (int k = 0; k < W.cols; ++k) {
        sum += W.data_fp32[i * W.cols + k] * x[k];
      }
      // Apply scale and ADD to the stream (Residual)
      // Simplified for 111M: if shapes match, we add.
      if (W.rows == x.size()) {
        next_x[i] = (sum * W.scale) + x[i];
      } else {
        next_x[i] = sum * W.scale;
      }
    }
    x = next_x;
  }
  output = x;
}
