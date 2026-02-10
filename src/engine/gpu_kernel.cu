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

// Global LUT in Constant Memory
__constant__ float d_LUT[4] = {0.0f, 1.0f, -1.0f, 0.0f};

// Kernel: MatMul with On-the-Fly 2-bit Unpacking
// Grid: (Rows / 16, 1)
// Block: (16, 16) - Or simple row-wise parallelization for now.
// Simple Kernel: Each thread computes one output element (row).
__global__ void
matmul_2bit_kernel(const uint8_t *__restrict__ W_packed, // (Rows, Cols/4)
                   const float *__restrict__ Input,      // (Cols)
                   float *__restrict__ Output,           // (Rows)
                   int Rows, int Cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Rows)
    return;

  float sum = 0.0f;
  int packed_cols = Cols / 4;

  // Loop through packed columns
  for (int k = 0; k < packed_cols; ++k) {
    uint8_t packed_byte = W_packed[row * packed_cols + k];

    // Unpack 4 weights per byte
    // Replicate logic: 0->0, 1->1, 2->-1, 3->0
    // (byte >> 6) & 3 etc.

    float w0 = d_LUT[(packed_byte >> 6) & 0x3];
    float w1 = d_LUT[(packed_byte >> 4) & 0x3];
    float w2 = d_LUT[(packed_byte >> 2) & 0x3];
    float w3 = d_LUT[(packed_byte >> 0) & 0x3];

    int input_idx = k * 4;
    sum += w0 * Input[input_idx + 0];
    sum += w1 * Input[input_idx + 1];
    sum += w2 * Input[input_idx + 2];
    sum += w3 * Input[input_idx + 3];
  }

  Output[row] = sum;
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

  matmul_2bit_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_W_packed, d_Input, d_Output, W.rows, W.cols);
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
