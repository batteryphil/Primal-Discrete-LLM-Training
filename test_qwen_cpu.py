import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_cpu_tps():
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    print(f"[*] Loading {model_name} on CPU (Normal FP16/BF16)...")
    
    # Force CPU and load with torch_dtype="auto" or float32 for CPU compatibility
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32 # CPU usually prefers float32 for stability/speed without specialized instructions
    )
    print(f"[*] Model loaded in {time.time() - start_load:.2f}s")

    prompt = "Write a quicksort function in Python."
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    print(f"[*] Generating 128 tokens (Average of 3 trials)...")
    
    # Warmup
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    results = []
    for i in range(3):
        start_gen = time.time()
        output = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False,
            use_cache=True
        )
        end_gen = time.time()
        
        total_tokens = output.shape[1] - inputs.input_ids.shape[1]
        duration = end_gen - start_gen
        results.append(total_tokens / duration)
        print(f"  Trial {i+1}: {total_tokens / duration:.2f} TPS")
    
    tps = sum(results) / len(results)
    
    print("\n" + "="*40)
    print(f"FINAL CPU BASELINE REPORT")
    print("="*40)
    print(f"Device:         CPU")
    print(f"Dtype:          Float32")
    print(f"Tokens Gen:     128")
    print(f"Average TPS:    {tps:.2f}")
    print("="*40)
    print(f"PRIME Comparison Goal: > 50.0 TPS (on GPU)")
    print("="*40)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nSample Output:\n{response[:200]}...")

if __name__ == "__main__":
    test_cpu_tps()
