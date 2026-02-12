#include "primal.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

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
__constant__ float c_LUT[16] = {0.0f,       1.0f,      -1.0f,      0.5f,
                                -0.5f,      0.333333f, -0.333333f, 0.2f,
                                -0.2f,      0.142857f, -0.142857f, 0.090909f,
                                -0.090909f, 0.076923f, -0.076923f, 0.0f};

// Explicit Residual Kernel (Legacy/Fallback)
__global__ void residual_add_kernel(float *x, const float *delta, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    x[i] += delta[i];
  }
}

// Phase 23.5: Safe 128-bit Kernel with Explicit Residual
__global__ void flash_primal_kernel(
    const uint8_t *__restrict__ W, const float *__restrict__ x,
    const float *__restrict__ residual_in, // Optional: Pass NULL if no residual
    float *__restrict__ out, float scale, int rows, int padded_cols,
    int real_cols // The actual size of x
) {
  // Force 16-byte alignment check (reinterpret_cast)
  const uint4 *W_vec = reinterpret_cast<const uint4 *>(W);

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  int lane = threadIdx.x & 31;

  // Iterate using PADDED columns for Weight alignment
  for (int k_tile = 0; k_tile < padded_cols; k_tile += 32) {

    // 1. Safe Input Load (Check against REAL cols)
    float tile_x_val = 0.0f;
    if (k_tile + lane < real_cols) {
      tile_x_val = x[k_tile + lane];
    }

    // 2. Vectorized Weight Load (Check against ROWS)
    if (row < rows) {
      int vec_idx = row * (padded_cols / 32) + (k_tile / 32);
      uint4 packed_vec = W_vec[vec_idx];

      uint32_t raw[4] = {packed_vec.x, packed_vec.y, packed_vec.z,
                         packed_vec.w};
      int current_k = 0;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t val = raw[i];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          uint8_t byte = (val >> (b * 8)) & 0xFF;

          // Decode High Nibble
          float w0 = c_LUT[(byte >> 4) & 0x0F] * scale;
          // Shuffle: Grab x from the specific lane 'current_k'
          float x0 = __shfl_sync(0xFFFFFFFF, tile_x_val, current_k);
          sum += w0 * x0;
          current_k++;

          // Decode Low Nibble
          float w1 = c_LUT[byte & 0x0F] * scale;
          float x1 = __shfl_sync(0xFFFFFFFF, tile_x_val, current_k);
          sum += w1 * x1;
          current_k++;
        }
      }
    }
  }

  // 3. Safe Residual Write
  if (row < rows) {
    float res = (residual_in != nullptr) ? residual_in[row] : 0.0f;
    out[row] = sum + res;
  }
}

// Legacy Matmul (Fallback for non-residual layers)
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
    float w0 = c_LUT[(packed_byte >> 4) & 0x0F];
    float w1 = c_LUT[(packed_byte >> 0) & 0x0F];
    int input_idx = k * 2;
    sum += w0 * Input[input_idx + 0];
    sum += w1 * Input[input_idx + 1];
  }
  Output[row] = sum * scale;
}

// Phase 20: Persistent Workspace Management
void PrimalEngine::init_workspace(size_t max_dim) {
  if (d_ping)
    cudaFree(d_ping);
  if (d_pong)
    cudaFree(d_pong);

  size_t alloc_floats = max_dim * 4;
  CHECK_CUDA(cudaMalloc(&d_ping, alloc_floats * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_pong, alloc_floats * sizeof(float)));
  workspace_size = alloc_floats;
  printf("[Primal] GPU Workspace Allocated: %zu floats per buffer\n",
         workspace_size);
}

void PrimalEngine::free_workspace() {
  if (d_ping)
    cudaFree(d_ping);
  if (d_pong)
    cudaFree(d_pong);
  d_ping = nullptr;
  d_pong = nullptr;
}

void PrimalEngine::gpu_forward(const std::vector<float> &input,
                               std::vector<float> &output) {
  if (weights.empty())
    return;

  if (!d_ping || !d_pong) {
    init_workspace(input.size() * 4);
  }

  CHECK_CUDA(cudaMemcpy(d_ping, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  float *src = d_ping;
  float *dst = d_pong;
  int current_size = input.size();

  for (const auto &W : weights) {
    int threads = 256;
    int blocks = (W.rows + threads - 1) / threads;

    if (W.rows == W.cols && W.rows == current_size) {
      // [FUSED PATH] Flash-Prime Kernel
      // W.d_data, x=src, res=src, out=dst, scale, rows, padded_cols=W.cols,
      // real_cols=current_size
      flash_primal_kernel<<<blocks, threads>>>(W.d_data, src, src, dst, W.scale,
                                               W.rows, W.cols, current_size);
      CHECK_CUDA(cudaDeviceSynchronize());
      std::swap(src, dst);

    } else {
      // [LEGACY PATH]
      matmul_4bit_kernel<<<blocks, threads>>>(W.d_data, src, dst, W.scale,
                                              W.rows, W.cols);
      CHECK_CUDA(cudaDeviceSynchronize());

      if (W.rows == current_size) {
        residual_add_kernel<<<blocks, threads>>>(dst, src, W.rows);
        CHECK_CUDA(cudaDeviceSynchronize());
      }
      current_size = W.rows;
      std::swap(src, dst);
    }
  }

  output.resize(current_size);
  CHECK_CUDA(cudaMemcpy(output.data(), src, current_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

// Phase 21: Zero-Copy Streaming Implementation
void PrimalEngine::gpu_forward_streamed(const float *h_input, float *h_output,
                                        int input_size) {
  if (weights.empty())
    return;

  if (stream == nullptr) {
    CHECK_CUDA(cudaStreamCreate((cudaStream_t *)&stream));
    printf("[Primal] CUDA Stream Created.\n");
  }
  cudaStream_t m_stream = (cudaStream_t)stream;

  if (!d_ping || !d_pong) {
    init_workspace(input_size * 4);
  }

  CHECK_CUDA(cudaMemcpyAsync(d_ping, h_input, input_size * sizeof(float),
                             cudaMemcpyHostToDevice, m_stream));

  float *src = d_ping;
  float *dst = d_pong;
  int current_size = input_size;

  for (const auto &W : weights) {
    int threads = 256;
    int blocks = (W.rows + threads - 1) / threads;

    if (W.rows == W.cols && W.rows == current_size) {
      // [FUSED PATH] Flash-Prime Kernel
      // Using Stream
      // W.d_data, x=src, res=src, out=dst, scale, rows, padded_cols=W.cols,
      // real_cols=current_size
      flash_primal_kernel<<<blocks, threads, 0, m_stream>>>(
          W.d_data, src, src, dst, W.scale, W.rows, W.cols, current_size);
    } else {
      // [LEGACY PATH]
      matmul_4bit_kernel<<<blocks, threads, 0, m_stream>>>(
          W.d_data, src, dst, W.scale, W.rows, W.cols);

      if (W.rows == current_size) {
        residual_add_kernel<<<blocks, threads, 0, m_stream>>>(dst, src, W.rows);
      }
      current_size = W.rows;
    }
    std::swap(src, dst);
  }

  CHECK_CUDA(cudaMemcpyAsync(h_output, src, current_size * sizeof(float),
                             cudaMemcpyDeviceToHost, m_stream));
  CHECK_CUDA(cudaStreamSynchronize(m_stream));
}
