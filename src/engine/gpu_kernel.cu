#include "primal.h"
#include <cstdio>
#include <cuda_runtime.h>


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
__constant__ float d_LUT[16] = {0.0f,       1.0f,      -1.0f,      0.5f,
                                -0.5f,      0.333333f, -0.333333f, 0.2f,
                                -0.2f,      0.142857f, -0.142857f, 0.090909f,
                                -0.090909f, 0.076923f, -0.076923f, 0.0f};

__global__ void matmul_4bit_kernel(const uint8_t *__restrict__ W_packed,
                                   const float *__restrict__ Input,
                                   float *__restrict__ Output, float scale,
                                   int Rows, int Cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Rows)
    return;

  float sum = 0.0f;
  int packed_cols = Cols / 2;

  for (int k = 0; k < packed_cols; ++k) {
    uint8_t packed_byte = W_packed[row * packed_cols + k];
    float w0 = d_LUT[(packed_byte >> 4) & 0x0F];
    float w1 = d_LUT[(packed_byte >> 0) & 0x0F];
    int input_idx = k * 2;
    sum += w0 * Input[input_idx + 0];
    sum += w1 * Input[input_idx + 1];
  }
  Output[row] = sum * scale;
}

void PrimalEngine::gpu_forward(const std::vector<float> &input,
                               std::vector<float> &output) {
  if (weights.empty())
    return;
  const Tensor &W = weights[0];

  uint8_t *d_W;
  float *d_In, *d_Out;
  size_t w_bytes = W.data_packed.size();
  size_t in_bytes = input.size() * sizeof(float);
  size_t out_bytes = W.rows * sizeof(float);

  CHECK_CUDA(cudaMalloc(&d_W, w_bytes));
  CHECK_CUDA(cudaMalloc(&d_In, in_bytes));
  CHECK_CUDA(cudaMalloc(&d_Out, out_bytes));

  CHECK_CUDA(
      cudaMemcpy(d_W, W.data_packed.data(), w_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_In, input.data(), in_bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (W.rows + threads - 1) / threads;
  matmul_4bit_kernel<<<blocks, threads>>>(d_W, d_In, d_Out, W.scale, W.rows,
                                          W.cols);
  CHECK_CUDA(cudaDeviceSynchronize());

  output.resize(W.rows);
  CHECK_CUDA(
      cudaMemcpy(output.data(), d_Out, out_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_W);
  cudaFree(d_In);
  cudaFree(d_Out);
}
