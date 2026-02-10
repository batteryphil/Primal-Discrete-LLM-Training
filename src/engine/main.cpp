#include "trinity.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>


int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [gpu]" << std::endl;
    return 1;
  }

  std::string model_path = argv[1];
  bool use_gpu = (argc > 2 && std::string(argv[2]) == "gpu");

  TrinityEngine engine;
  engine.use_gpu = use_gpu;

  // Load Model
  auto start_load = std::chrono::high_resolution_clock::now();
  engine.load(model_path);
  auto end_load = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> load_duration = end_load - start_load;
  std::cout << "[Trinity] Load Time: " << load_duration.count() << " seconds"
            << std::endl;

  // Create Dummy Input (Size 2048 matches the dummy layer in cpu_kernel.cpp)
  // In a real scenario, this would be embeddings from a tokenizer.
  std::vector<float> input(2048, 1.0f);
  std::vector<float> output;

  // Warmup
  std::cout << "[Trinity] Warming up..." << std::endl;
  engine.forward(input, output);

  // Benchmark
  int num_runs = 10;
  std::cout << "[Trinity] benchmarking " << num_runs << " runs..." << std::endl;

  auto start_infer = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_runs; ++i) {
    engine.forward(input, output);
  }
  auto end_infer = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> infer_duration = end_infer - start_infer;

  double avg_time = infer_duration.count() / num_runs;
  std::cout << "[Trinity] Average Inference Time: " << avg_time * 1000 << " ms"
            << std::endl;

  if (!output.empty()) {
    std::cout << "[Trinity] Output[0]: " << output[0] << std::endl;
  }

  return 0;
}
