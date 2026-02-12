#include "primal.h"
#include <cstdlib>
#include <iostream>
#include <vector>


// This function is linked ONLY when building the CPU-only version
// (primal_cpu.exe). It acts as a safety catch if the user tries to run GPU mode
// on a CPU build.
void PrimalEngine::gpu_forward(const std::vector<float> &input,
                               std::vector<float> &output) {
  std::cerr
      << "\n[CRITICAL ERROR] GPU Mode Requested, but this binary is CPU-ONLY."
      << std::endl;
  std::cerr << "Please run 'primal_gpu.exe' instead of 'primal_cpu.exe'."
            << std::endl;
  exit(1);
}
