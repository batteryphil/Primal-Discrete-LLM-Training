#include "primal.h"
#include <vector>

// We use extern "C" to prevent name mangling so Python can find the functions
extern "C" {
// 1. Create the Engine Instance
__declspec(dllexport) PrimalEngine *create_engine(const char *model_path,
                                                  bool use_gpu) {
  PrimalEngine *engine = new PrimalEngine();
  engine->use_gpu = use_gpu;
  // In the DLL context, paths might be relative to the caller.
  // We'll trust the Python script to pass the right path.
  engine->load(model_path);
  return engine;
}

// 2. Perform Inference
// input_size should match the hidden dimension (e.g., 768)
__declspec(dllexport) void run_inference(PrimalEngine *engine, float *input,
                                         float *output, int input_size) {
  if (!engine)
    return;
  std::vector<float> in_vec(input, input + input_size);
  std::vector<float> out_vec;

  // This runs the full residual stream forward pass we just verified
  engine->forward(in_vec, out_vec);

  // Copy the result back to the Python pointer
  // The output size depends on the last layer (Vocabulary Size approx 50257)
  for (size_t i = 0; i < out_vec.size(); ++i) {
    output[i] = out_vec[i];
  }
}

// 3. Cleanup
__declspec(dllexport) void destroy_engine(PrimalEngine *engine) {
  if (engine)
    delete engine;
}

// 4. Telemetry (Placeholder for Python NVML)
__declspec(dllexport) void get_gpu_stats(int *temp, float *power, int *vram) {
  if (temp)
    *temp = -1;
  if (power)
    *power = -1.0f;
  if (vram)
    *vram = -1;
  // Real telemetry is handled by primal_bench.py via nvidia-smi
}

// Optimization: Zero-Copy Pointer Management (Phase 21)
__declspec(dllexport) void run_inference_fast(PrimalEngine *engine,
                                              float *h_input, float *h_output,
                                              int input_size) {
  if (!engine)
    return;

  // Use the engine's internal device buffers directly via Async Stream
  engine->gpu_forward_streamed(h_input, h_output, input_size);
}
}
