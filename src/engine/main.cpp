#include "primal.h"
#include <chrono>
#include <iostream>


int main(int argc, char **argv) {
  bool use_gpu = (argc > 1 && std::string(argv[1]) == "gpu");
  std::cout << "Starting Project PRIMAL V3.0.0 [Mode: "
            << (use_gpu ? "GPU" : "CPU") << "]" << std::endl;

  PrimalEngine engine;
  engine.use_gpu = use_gpu;
  engine.load("dummy_path");

  std::vector<float> input(2048, 1.0f);
  std::vector<float> output;

  // Warmup
  engine.forward(input, output);

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; ++i)
    engine.forward(input, output);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Avg Time: " << (elapsed.count() / 10.0) * 1000 << " ms"
            << std::endl;
  std::cout << "Output[0]: " << output[0] << std::endl;
  return 0;
}
