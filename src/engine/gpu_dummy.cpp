#include "primal.h"
#include <iostream>
#include <vector>

// This is linked only when building for CPU to satisfy the linker.
// The GPU build links gpu_kernel.cu instead.
void PrimalEngine::gpu_forward(const std::vector<float> &input,
                               std::vector<float> &output) {
  std::cerr << "Error: GPU support was not compiled into this executable."
            << std::endl;
  exit(1);
}
