#include "primal.h"
#include <algorithm>
#include <iostream>
#include <vector>


int main(int argc, char **argv) {
  bool use_gpu = (argc > 1 && std::string(argv[1]) == "gpu");
  std::cout << "Starting Project PRIMAL V3.0.0 [Mode: "
            << (use_gpu ? "GPU" : "CPU") << "]" << std::endl;

  PrimalEngine engine;
  engine.use_gpu = use_gpu;
  engine.load("model.primal");

  if (engine.weights.empty()) {
    std::cout << "Failed to load model." << std::endl;
    return 1;
  }

  // Cerebras-111M Hidden Dim is 768
  // Note: We are simulating a 'Residual Stream' input at the hidden layer level
  // skipping the embedding lookup logic inside the engine for now.
  // If the first layer is WTE, it might reshape.
  std::vector<float> input(768, 0.5f);
  std::vector<float> output;

  std::cout << "[Full Pipeline] Running " << engine.weights.size()
            << " Layers..." << std::endl;
  engine.forward(input, output);

  // Find the 'Argmax' (The most likely token)
  if (!output.empty()) {
    auto max_it = std::max_element(output.begin(), output.end());
    int token_id = std::distance(output.begin(), max_it);

    std::cout << "Pipeline Complete." << std::endl;
    std::cout << "Output Size: " << output.size() << std::endl;
    std::cout << "Top Predicted Token ID: " << token_id << std::endl;
  } else {
    std::cout << "Error: No output generated." << std::endl;
  }

  return 0;
}
