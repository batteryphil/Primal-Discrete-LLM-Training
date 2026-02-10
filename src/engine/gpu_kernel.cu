#include "trinity.h"
#include <cstdio>
#include <cuda_runtime.h>

// CUDA Error Check Helper
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err),           \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  }

// Global LUT in Constant Memory (16 values)
// Grid: V3.0.0 "Prime Rich"
__constant__ float d_LUT[16] = {
    0.0f,                  // 0x0
    1.0f,      -1.0f,      // 0x1, 0x2
    0.5f,      -0.5f,      // 0x3, 0x4
    0.333333f, -0.333333f, // 0x5, 0x6
    0.2f,      -0.2f,      // 0x7, 0x8
    0.142857f, -0.142857f, // 0x9, 0xA
    0.090909f, -0.090909f, // 0xB, 0xC
    0.076923f, -0.076923f, // 0xD, 0xE
    0.0f                   // 0xF
};

// Kernel: MatMul with On-the-Fly 4-bit Unpacking
// Grid: (Rows / 16, 1)
// Block: (16, 16)
__global__ void
matmul_4bit_kernel(const uint8_t *__restrict__ W_packed, // (Rows, Cols/2)
                   const float *__restrict__ Input,      // (Cols)
                   float *__restrict__ Output,           // (Rows)
                   float scale,                          // Scale factor
                   int Rows, int Cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Rows)
    return;

  float sum = 0.0f;
  // 4-bit packing = 2 weights per byte
  int packed_cols = Cols / 2;

  // Loop through packed columns
  for (int k = 0; k < packed_cols; ++k) {
    uint8_t packed_byte = W_packed[row * packed_cols + k];

    // Unpack 2 weights per byte (High Nibble, Low Nibble)
    float w0 = d_LUT[(packed_byte >> 4) & 0x0F]; // High
    float w1 = d_LUT[(packed_byte >> 0) & 0x0F]; // Low

    int input_idx = k * 2;
    sum += w0 * Input[input_idx + 0];
    sum += w1 * Input[input_idx + 1];
  }

  Output[row] = sum * scale;
}

void TrinityEngine::gpu_forward(const std::vector<float> &input,
                                std::vector<float> &output) {
  if (weights.empty())
    return;
  const Tensor &W = weights[0]; // Test first layer

  // Allocate Device Memory
  uint8_t *d_W_packed;
  float *d_Input;
  float *d_Output;

  size_t w_bytes = W.data_packed.size() * sizeof(uint8_t);
  size_t in_bytes = input.size() * sizeof(float);
  size_t out_bytes = W.rows * sizeof(float);

  CHECK_CUDA(cudaMalloc(&d_W_packed, w_bytes));
  CHECK_CUDA(cudaMalloc(&d_Input, in_bytes));
  CHECK_CUDA(cudaMalloc(&d_Output, out_bytes));

  // Copy Data
  CHECK_CUDA(cudaMemcpy(d_W_packed, W.data_packed.data(), w_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_Input, input.data(), in_bytes, cudaMemcpyHostToDevice));

  // Launch Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (W.rows + threadsPerBlock - 1) / threadsPerBlock;

  matmul_4bit_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_W_packed, d_Input, d_Output, W.scale, W.rows, W.cols);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy Back
  output.resize(W.rows);
  CHECK_CUDA(
      cudaMemcpy(output.data(), d_Output, out_bytes, cudaMemcpyDeviceToHost));

  // Free
  cudaFree(d_W_packed);
  cudaFree(d_Input);
  cudaFree(d_Output);
}
