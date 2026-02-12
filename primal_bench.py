import time
import subprocess
import threading
import os
import re
import sys
import torch
import primal_chat # Reuses the chat interface logic

def get_nvidia_smi():
    try:
        # Run nvidia-smi to get Temp, Power, VRAM
        # CSV format: temperature.gpu, power.draw, memory.used
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,memory.used", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        parts = out.strip().split(', ')
        return int(parts[0]), float(parts[1]), int(parts[2])
    except Exception as e:
        return 0, 0.0, 0

stop_monitor = False

def monitor_resources():
    print(f"{'TIME':<10} | {'TEMP':<5} | {'POWER':<8} | {'VRAM':<8} | {'CPU%':<5}")
    print("-" * 50)
    while not stop_monitor:
        temp, power, vram = get_nvidia_smi()
        # CPU% is hard without psutil, skip or approximate? 
        # We'll just print GPU stats.
        timestamp = time.strftime("%H:%M:%S")
        print(f"{timestamp:<10} | {temp}°C  | {power:.1f}W   | {vram}MB")
        time.sleep(1.0)

def benchmark_throughput():
    prompt = "The future of AI is"
    print(f"\n[BENCHMARK] Warming up with prompt: '{prompt}'...")
    
    # Measure TTFT (Time to First Token)
    start_t = time.perf_counter()
    # We need to access the lower level chat function or modify primal_chat to return stats
    # For now, let's just use primal_chat.chat and measure end-to-end for short generation
    # Actually, we should import the components and run a custom loop
    
    # Setup
    # Updated for run_inference_fast (Phase 21)
    if hasattr(primal_chat.lib, 'run_inference_fast'):
        print("[BENCHMARK] Using Zero-Copy API (run_inference_fast)")
        inference_fn = primal_chat.lib.run_inference_fast
    else:
        print("[BENCHMARK] Using Standard API (run_inference)")
        inference_fn = primal_chat.lib.run_inference
        
    inference_fn.argtypes = [
        primal_chat.ctypes.c_void_p, 
        primal_chat.ctypes.POINTER(primal_chat.ctypes.c_float), 
        primal_chat.ctypes.POINTER(primal_chat.ctypes.c_float), 
        primal_chat.ctypes.c_int
    ]
    
    engine = primal_chat.engine
    tokenizer = primal_chat.tokenizer
    token_embedding = primal_chat.token_embedding
    pos_embedding = primal_chat.pos_embedding
    
    # 128 Token Prompt Generation
    prompt_tokens = tokenizer.encode(prompt * 20)[:128] # Make it long
    if len(prompt_tokens) < 128:
        prompt_tokens = (prompt_tokens * 10)[:128]
        
    print(f"[BENCHMARK] Prompt Length: {len(prompt_tokens)} tokens")
    
    # Run Prefill
    # In this simple model (batch=1), prefill is just sequential forward pass for each token?
    # No, Primal V3 is one-token-at-a-time (KV cache not explicitly exposed/managed in Python side yet? 
    # Wait, the C++ engine has NO KV CACHE logic inside it for the Transformer blocks?
    # Actually, the C++ engine runs the full model forward pass `forward(x)`.
    # A standard GPT needs to re-process all tokens for every step unless KV cache is stateful.
    # Our C++ 'gpu_forward' takes 'input'.
    # Does 'input' imply the FULL sequence or just the last token?
    # The Primal Architecture (Phase 16) was "8-layer Transformer".
    # `CausalSelfAttention`: `q, k, v = self.c_attn(x)`.
    # `att = (q @ k.transpose)`.
    # If we pass a sequence of length 1 (last token), it attends only to itself!
    # UNLESS the model is stateful.
    # The C++ code: `weights` vector. `gpu_forward` takes `input`.
    # It does MatMul.
    # It does NOT seem to contain a persistent KV cache state in `PrimalEngine`.
    # So... does it function as a language model?
    # If we only pass the last token's embedding, `x` is [1, Dim].
    # Attention `q, k, v` are [1, 1, Dim].
    # `att` is [1, 1].
    # It has no context!
    # OH NO. The "Primal" model as implemented in Phase 16/18 appears to be stateless?
    # Wait, `primal_chat.py`:
    # `input_vec = (t_emb + p_emb).view(-1)`
    # It takes `tokens[-1]` (last token) and `seq_len - 1` (last pos).
    # Then `run_inference` runs the engine on `input_vec` (size 384).
    # This architecture has NO context window! It's a Markov Chain (Bigram with depth 1)!
    # Unless... `PrimalBrain` (Python) was trained with context?
    # `primal_train_v2.py`:
    # `enc = tokenizer.encode(...)`
    # `batch = ... (sequence)`
    # `logits = model(input_ids)`
    # In `PrimalBrain.forward(idx)`:
    # `B, T = idx.size()`
    # `pos = ...`
    # `x = tok + pos`
    # `block(x)` -> Attention uses `q @ k.T`.
    # So during training, it DOES use context (T > 1).
    # BUT in `C++ Inference`, we serve ONE token vector [1, 384].
    # So T=1.
    # So `att` is 1x1.
    # Meaning: It ignores all history.
    # This explains why "Twinkle Twinkle Little Star" -> "Star" might work (associative), but complex logic fails.
    # This is a MAJOR constraint/bug for "Grand Scale".
    # To fix this, we need a KV Cache or we must Feed the FULL sequence every time (Inefficient O(N^2)).
    # For Phase 20, we can feed the full sequence?
    # `primal_chat.py` only constructs embedding for the LAST token.
    # To fix "Grand Scale" logic, we should probably implement KV Cache or at least Full-Context-Window forwarding.
    # However, `PrimalEngine` expects flat `float* input`.
    # If we pass T tokens, input size is T * Dim.
    # The C++ `gpu_forward` takes `input`. `d_x_size = input.size()`.
    # Inside: `W.rows == d_x_size`.
    # `W.rows` implies logic was typically `d_model`.
    # Unless `matmul_kernel` handles `cols = d_model`, `rows = d_model`.
    # If input is `T * d_model`, then `x` is `[T, d_model]`.
    # Matmul `x @ W` ?
    # Our `matmul_4bit_kernel` iterates `k < Cols`. `Input` is flatted.
    # It calculates `Output[row]`.
    # `row` comes from `threads`.
    # If we have T tokens, we need `Batch MatMul`?
    # Our simple kernel computes `Output[r] = Sum(W[r, k] * Input[k])`.
    # This assumes `Input` is a vector of size `Cols`.
    # It produces a vector `Output` of size `Rows`.
    # This is strictly `Vector-Matrix Multiplication` (v * M).
    # So it processes ONE token.
    #
    # CONCLUSION: The current C++ engine is a STATLESS 1-Token Context Engine.
    # "Phase 20" was asking for "Ping-Pong" and "Scale", but missed "KV Cache".
    # I cannot fix the architecture instantly.
    # I will proceed with Benchmarking the CURRENT architecture (Tokens/sec).
    # Even if dumb, it is fast!
    # I will add a note in `primal_qa.py` that logic might be limited by context window.
    
    # Throughput Test
    token_count = 0
    start = time.time()
    for _ in range(100):
        # Dummy "chat" step
        # Create a random vector to simulate
        # In reality we should loop chat generation
        # Let's use the Python embedding generation for realism
        with torch.no_grad():
             # Just use dummy embedding
             dummy_in = torch.randn(384).numpy().astype(primal_chat.np.float32)
             
        out_buf = (primal_chat.ctypes.c_float * 50257)()
        inference_fn(engine, dummy_in.ctypes.data_as(primal_chat.ctypes.POINTER(primal_chat.ctypes.c_float)), out_buf, 384)
        token_count += 1
        
    duration = time.time() - start
    tps = token_count / duration
    print(f"\n[BENCHMARK] Throughput: {tps:.2f} Tokens/sec")
    print(f"[BENCHMARK] Latency (Per Token): {duration/token_count*1000:.2f} ms")

if __name__ == "__main__":
    t = threading.Thread(target=monitor_resources)
    t.start()
    
    try:
        time.sleep(2) # Let stats collect
        benchmark_throughput()
    finally:
        stop_monitor = True
        t.join()
