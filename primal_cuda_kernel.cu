#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


template <typename scalar_t>
__global__ void tandem_forward_kernel(const uint8_t *__restrict__ base_idx,
                                      const uint8_t *__restrict__ fine_idx,
                                      const scalar_t *__restrict__ scale,
                                      const scalar_t *__restrict__ lut,
                                      scalar_t *__restrict__ output,
                                      int num_rows, int num_cols) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < num_rows && col < num_cols) {
    int idx = row * num_cols + col;
    int lut_idx = (static_cast<int>(base_idx[idx]) * 256) +
                  static_cast<int>(fine_idx[idx]);
    output[idx] = lut[lut_idx] * scale[row]; // scale is [num_rows, 1]
  }
}

template <typename scalar_t>
__global__ void tandem_backward_kernel(
    const scalar_t *__restrict__ grad_output,
    const uint8_t *__restrict__ base_idx, const uint8_t *__restrict__ fine_idx,
    const scalar_t *__restrict__ scale, const scalar_t *__restrict__ lut,
    int16_t *__restrict__ vote_buffer, scalar_t *__restrict__ grad_scale,
    float grad_max, int num_rows, int num_cols) {

  int row = blockIdx.x;
  int tid = threadIdx.x;

  scalar_t scale_val = scale[row];
  // Equivalent to sign() logic
  float sign_scale =
      (scale_val > 0.0f) ? 1.0f : ((scale_val < 0.0f) ? -1.0f : 0.0f);

  // Use shared memory for reduction
  extern __shared__ unsigned char shared_mem[];
  float *sdata = reinterpret_cast<float *>(shared_mem);

  float thread_sum = 0.0f;

  for (int col = tid; col < num_cols; col += blockDim.x) {
    int idx = row * num_cols + col;
    float g_out = static_cast<float>(grad_output[idx]);

    int lut_idx = (static_cast<int>(base_idx[idx]) * 256) +
                  static_cast<int>(fine_idx[idx]);
    float w_proxy = static_cast<float>(lut[lut_idx]);

    thread_sum += g_out * w_proxy;

    // Vote Injection
    float g_normed = g_out / grad_max;
    int32_t pressure = static_cast<int32_t>(g_normed * 100.0f * sign_scale);
    int32_t current_vote = static_cast<int32_t>(vote_buffer[idx]);

    int32_t new_vote = current_vote + pressure;
    if (new_vote > 32760)
      new_vote = 32760;
    else if (new_vote < -32760)
      new_vote = -32760;

    vote_buffer[idx] = static_cast<int16_t>(new_vote);
  }

  sdata[tid] = thread_sum;
  __syncthreads();

  // Reduction
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    grad_scale[row] = static_cast<scalar_t>(sdata[0]);
  }
}

torch::Tensor tandem_forward_cuda(torch::Tensor base_idx,
                                  torch::Tensor fine_idx, torch::Tensor scale,
                                  torch::Tensor lut) {

  auto output = torch::empty_like(base_idx, scale.options());

  int num_rows = base_idx.size(0);
  int num_cols = base_idx.size(1);

  dim3 threads(32, 32);
  dim3 blocks((num_cols + threads.x - 1) / threads.x,
              (num_rows + threads.y - 1) / threads.y);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      scale.scalar_type(), "tandem_forward_cuda", ([&] {
        tandem_forward_kernel<scalar_t><<<blocks, threads>>>(
            base_idx.data_ptr<uint8_t>(), fine_idx.data_ptr<uint8_t>(),
            scale.data_ptr<scalar_t>(), lut.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), num_rows, num_cols);
      }));

  return output;
}

torch::Tensor tandem_backward_cuda(torch::Tensor grad_output,
                                   torch::Tensor base_idx,
                                   torch::Tensor fine_idx, torch::Tensor scale,
                                   torch::Tensor lut, torch::Tensor vote_buffer,
                                   float grad_max) {

  auto grad_scale = torch::empty_like(scale);
  int num_rows = base_idx.size(0);
  int num_cols = base_idx.size(1);

  int threads_per_block = 512;
  int shared_mem_size = threads_per_block * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      scale.scalar_type(), "tandem_backward_cuda", ([&] {
        tandem_backward_kernel<scalar_t>
            <<<num_rows, threads_per_block, shared_mem_size>>>(
                grad_output.data_ptr<scalar_t>(), base_idx.data_ptr<uint8_t>(),
                fine_idx.data_ptr<uint8_t>(), scale.data_ptr<scalar_t>(),
                lut.data_ptr<scalar_t>(), vote_buffer.data_ptr<int16_t>(),
                grad_scale.data_ptr<scalar_t>(), grad_max, num_rows, num_cols);
      }));

  return grad_scale;
}
