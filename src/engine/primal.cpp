#include "primal.h"
#include <cuda_runtime.h> // Required for cudaMalloc/cudaMemcpy
#include <fstream>
#include <iostream>
#include <vector>


void PrimalEngine::load(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "Error: Could not open file " << path << std::endl;
    return;
  }

  char magic[4];
  f.read(magic, 4);
  if (std::string(magic, 4) != "PRML") {
    std::cerr << "Error: Invalid magic header. Expected 'PRML'." << std::endl;
    return;
  }

  while (f.peek() != EOF) {
    Tensor t;
    uint32_t name_len;
    f.read((char *)&name_len, 4);
    if (f.eof())
      break;

    std::vector<char> name(name_len);
    f.read(name.data(), name_len);
    t.name = std::string(name.data(), name_len);

    uint32_t data_size;

    // Read Dimensions & Scale
    f.read((char *)&t.rows, 4);
    f.read((char *)&t.cols, 4);
    f.read((char *)&t.scale, 4);
    f.read((char *)&data_size, 4);

    // FILTER: Skip Embedding Layers (wte, wpe)
    if (t.name.find("embedding") != std::string::npos) {
      std::cout << "[Primal] Skipping Embedding Layer: " << t.name
                << " (Handled by Host)" << std::endl;
      // Skip Data
      f.seekg(data_size, std::ios::cur);
      continue;
    }

    if (use_gpu) {
      // [PHASE 20] Chunked Loading: Disk -> Temporary CPU Buffer -> VRAM
      // This saves System RAM by never holding the whole model.

      // 1. Allocate Temp Buffer
      std::vector<uint8_t> temp_buffer(data_size);
      f.read((char *)temp_buffer.data(), data_size);

      // 2. Allocate VRAM
      cudaError_t err = cudaMalloc(&t.d_data, data_size);
      if (err != cudaSuccess) {
        std::cerr << "CUDA Error (Malloc): " << cudaGetErrorString(err)
                  << std::endl;
        exit(1);
      }

      // 3. Move to GPU
      err = cudaMemcpy(t.d_data, temp_buffer.data(), data_size,
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        std::cerr << "CUDA Error (Memcpy): " << cudaGetErrorString(err)
                  << std::endl;
        exit(1);
      }

      // temp_buffer is freed here automatically
      t.data_packed.clear(); // Ensure CPU vector is empty

    } else {
      // Legacy CPU Mode
      t.data_packed.resize(data_size);
      f.read((char *)t.data_packed.data(), data_size);

      // Dequantize for CPU? (Or implemented JIT)
      // Original code dequantized to fp32 here, but let's stick to packed for
      // now or follow original pattern The original pattern actually
      // dequantized to data_fp32. For standard "Grand Scale", we probably
      // wouldn't run on CPU, but let's correct this. The original snippet shows
      // loop to fill data_fp32.
      t.data_fp32.resize((size_t)t.rows * (size_t)t.cols);
      for (size_t i = 0; i < data_size; ++i) {
        uint8_t byte = t.data_packed[i];
        size_t idx = i * 2;
        if (idx + 1 < t.data_fp32.size()) {
          t.data_fp32[idx + 0] =
              LUT[(byte >> 4) & 0x0F] * t.scale; // Added Scale application
          t.data_fp32[idx + 1] = LUT[byte & 0x0F] * t.scale;
        }
      }
    }

    weights.push_back(t);
  }
  std::cout << "[Primal] Loaded " << weights.size()
            << " computational layers from " << path << std::endl;

  // Calculate Max Width for Workspace
  size_t max_dim = 0;
  for (const auto &w : weights) {
    if (w.rows > max_dim)
      max_dim = w.rows;
    if (w.cols > max_dim)
      max_dim = w.cols;
  }
  if (use_gpu)
    init_workspace(max_dim);
}
