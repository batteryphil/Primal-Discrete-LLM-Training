import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_gpu_tps():
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("[!] CUDA not available. Skipping GPU test.")
        return

    print(f"[*] Loading {model_name} on GPU ({device}) in FP16...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    start_load = time.time()
    
    # Load in FP16 on GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print(f"[*] Model loaded in {time.time() - start_load:.2f}s")

    prompt = "Write a quicksort function in Python."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"[*] Generating 128 tokens (Average of 3 trials)...")
    
    # Warmup
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()
    
    results = []
    for i in range(3):
        start_gen = time.time()
        output = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False,
            use_cache=True
        )
        torch.cuda.synchronize()
        end_gen = time.time()
        
        total_tokens = output.shape[1] - inputs.input_ids.shape[1]
        duration = end_gen - start_gen
        results.append(total_tokens / duration)
        print(f"  Trial {i+1}: {total_tokens / duration:.2f} TPS")
    
    tps = sum(results) / len(results)
    
    print("\n" + "="*40)
    print(f"GPU NORMAL PERFORMANCE REPORT")
    print("="*40)
    print(f"Device:         {torch.cuda.get_device_name(0)}")
    print(f"Dtype:          Float16")
    print(f"Tokens Gen:     128")
    print(f"Average TPS:    {tps:.2f}")
    print("="*40)

if __name__ == "__main__":
    test_gpu_tps()
