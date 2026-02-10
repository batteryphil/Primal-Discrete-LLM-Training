import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import struct
import os
import argparse
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_prime_grid():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151]
    reciprocals = [1.0/p for p in primes]
    tails = [1.0, 1.5, 2.0, 3.0]
    full_grid = sorted([0.0] + reciprocals + tails + [-r for r in reciprocals] + [-t for t in tails])
    return torch.tensor(full_grid, device=DEVICE, dtype=torch.float16)

class PrimeQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, grid):
        sigma = weight.std()
        if sigma == 0: sigma = 1e-6
        w_norm = weight / sigma
        w_flat = w_norm.reshape(-1)
        w_quant_flat = torch.zeros_like(w_flat)
        grid_flat = grid.reshape(1, -1)
        # Optimize chunk size for CPU L3 cache (avoid excessive thrashing)
        # GPU handles 1M chunks fine, CPU needs smaller chunks (e.g., 256k or 32k)
        if weight.device.type == 'cpu':
            chunk_size = 32 * 1024 
        else:
            chunk_size = 1024 * 1024

        for i in range(0, w_flat.numel(), chunk_size):
            chunk = w_flat[i : i + chunk_size].reshape(-1, 1)
            dist = torch.abs(chunk - grid_flat)
            nearest_idx = torch.argmin(dist, dim=1)
            w_quant_flat[i : i + chunk_size] = grid[nearest_idx]
        return w_quant_flat.view(weight.shape) * sigma
    @staticmethod
    def backward(ctx, grad_output): return grad_output, None

class PrimeLinear(nn.Linear):
    def __init__(self, original_layer):
        super().__init__(original_layer.in_features, original_layer.out_features, original_layer.bias is not None)
        with torch.no_grad():
            self.weight.data = original_layer.weight.data.clone()
            if original_layer.bias is not None: self.bias.data = original_layer.bias.data.clone()
            self.original_std = original_layer.weight.std().item()
        self.register_buffer('grid', get_prime_grid())
        self.is_quantized = False # Freeze-Quant flag

    def forward(self, x):
        if not self.is_quantized:
            with torch.no_grad():
                # Perform quantization once and freeze
                w_quant = PrimeQuantFunction.apply(self.weight, self.grid)
                current_std = w_quant.std()
                if current_std > 0:
                    scale = self.original_std / current_std.item()
                    w_quant = w_quant * scale
                self.weight.copy_(w_quant)
                self.is_quantized = True
        
        return F.linear(x, self.weight, self.bias)

def unpack_ternary_weights(packed_bytes, original_shape):
    numel = torch.Size(original_shape).numel()
    packed_tensor = torch.from_numpy(from_buffer(packed_bytes, dtype='uint8')).to(torch.int16)
    
    # Unpack 4 weigths per byte: (w0 << 6) | (w1 << 4) | (w2 << 2) | w3
    w0 = (packed_tensor >> 6) & 0x03
    w1 = (packed_tensor >> 4) & 0x03
    w2 = (packed_tensor >> 2) & 0x03
    w3 = packed_tensor & 0x03
    
    w_unpacked = torch.stack([w0, w1, w2, w3], dim=1).flatten()
    # Map back from {0, 1, 2} to {-1, 0, 1}
    w_sign = w_unpacked[:numel].to(torch.float16) - 1.0
    return w_sign.view(original_shape)

def from_buffer(b, dtype):
    import numpy as np
    return np.frombuffer(b, dtype=dtype)

def load_trinity_bin(model, bin_path):
    print(f"Unpacking Trinity Binary: {bin_path}")
    with open(bin_path, "rb") as f:
        header = f.read(4)
        if header != b'TRIN':
            raise ValueError("Invalid Trinity Model File (Missing TRIN header)")
        
        for name, module in tqdm(model.named_modules(), desc="Restoring Grid"):
            if isinstance(module, PrimeLinear):
                len_packed = struct.unpack('I', f.read(4))[0]
                packed_data = f.read(len_packed)
                with torch.no_grad():
                    # Restore weights into the FP16 shadow weight buffer
                    unpacked = unpack_ternary_weights(packed_data, module.weight.shape)
                    module.weight.copy_(unpacked.to(DEVICE))
    print("Restore Complete.")

def main():
    parser = argparse.ArgumentParser(description="Trinity-1.58bit Inference Engine")
    parser.add_argument("--model", type=str, required=True, help="Path to trinity_1.58bit_packed.bin")
    parser.add_argument("--prompt", type=str, default="Instruction: What is the capital of France?\nResponse: ", help="Input prompt")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu or cuda)")
    args = parser.parse_args()

    global DEVICE
    if args.device:
        DEVICE = args.device
    
    print(f"--- PROJECT TRINITY: INFERENCE ENGINE (V1.0.0) [Device: {DEVICE}] ---")
    
    # CPU Optimization: FP16 is often slower on CPUs due to emulation. Use FP32.
    if DEVICE == "cpu":
        print("   [O] CPU Detected: Forcing Float32 for AVX2/AVX512 optimization.")
        dtype = torch.float32
        # Auto-tune threads (physical cores usually best for matmul)
        import os
        cpu_count = os.cpu_count()
        if cpu_count:
            phys_cores = max(1, cpu_count // 2)
            torch.set_num_threads(phys_cores)
            print(f"   [O] CPU Threads set to: {phys_cores}")
    else:
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Loading Base Architecture...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(DEVICE)
    
    # Inject Prime Architecture
    replacements = []
    for name, module in model.named_modules():
         # Every Linear layer was packed by pack.py, so every one must be replaced
         if isinstance(module, nn.Linear):
             replacements.append((name, module))
    
    print(f"Injecting Prime Grid into {len(replacements)} layers...")
    for name, module in replacements:
        parent = model.get_submodule(name.rsplit('.', 1)[0] if '.' in name else '')
        setattr(parent, name.rsplit('.', 1)[-1], PrimeLinear(module).to(DEVICE))

    # Load and Unpack Binary
    load_trinity_bin(model, args.model)
    
    # Optimization: Pre-Freeze Weights
    # The loaded weights are {-1, 0, 1}. We can skip the expensive PrimeQuantFunction search
    # and directly apply the scale matching. This makes initialization instant.
    print("Optimization: Pre-scaling weights to avoid search overhead...")
    for name, module in model.named_modules():
        if isinstance(module, PrimeLinear):
             with torch.no_grad():
                 # 1. Calculate Scale
                 current_std = module.weight.std()
                 if current_std > 0:
                     scale = module.original_std / current_std.item()
                     # 2. Apply Scale In-Place
                     module.weight.mul_(scale)
                 # 3. Mark as Frozen (skips forward pass search)
                 module.is_quantized = True

    model.eval()

    print(f"\nPrompt: {args.prompt}")
    tokens = tokenizer(args.prompt, return_tensors="pt").to(DEVICE)
    
    print("Generating (max 32 tokens)...")
    import time
    with torch.no_grad():
        try:
            # First forward pass (Freeze-Quant trigger)
            tokens = tokens.to(DEVICE)
            model(**tokens)
            
            # CPU Optimization: Dynamic Quantization (Int8)
            # This reduces memory bandwidth by 4x vs FP32 and 2x vs FP16
            if DEVICE == "cpu":
                print("   [O] CPU Detected: Applying Dynamic Quantization (Int8)...")
                
                # 1. Convert PrimeLinear back to vanilla nn.Linear (standardize for quantizer)
                # Note: The weights are already "Frozen" and scaled from the forward pass above
                replacements = []
                for name, module in model.named_modules():
                    if isinstance(module, PrimeLinear):
                        # Verify it was frozen
                        if not module.is_quantized:
                             # Should not happen with Pre-Freeze, but safe fallback
                             pass
                        replacements.append((name, module))
                
                print(f"   [O] Converting {len(replacements)} layers to Int8 for CPU execution...")
                
                for name, module in replacements:
                    new_layer = nn.Linear(module.in_features, module.out_features, module.bias is not None)
                    with torch.no_grad():
                        new_layer.weight.copy_(module.weight) # Copy frozen/scaled weights
                        if module.bias is not None: new_layer.bias.copy_(module.bias)
                    
                    # Replace in parent
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                        setattr(parent, name.rsplit('.', 1)[-1], new_layer)

                # 2. Apply Dynamic Quantization
                model = torch.ao.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
                print("   [O] Int8 Quantization Complete. Running Inference...")
                print("   [O] Quantization Complete. Running Inference...")
            
            start_time = time.time()
            output = model.generate(
                **tokens, 
                max_new_tokens=32, 
                temperature=0.7, 
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            end_time = time.time()
            
            gen_duration = end_time - start_time
            num_tokens = output.shape[1] - tokens.input_ids.shape[1]
            tps = num_tokens / gen_duration if gen_duration > 0 else 0
            
            print(f"\nResult: {tokenizer.decode(output[0], skip_special_tokens=True)}")
            print(f"\n--- SPEED TEST ---")
            print(f"Tokens Generated: {num_tokens}")
            print(f"Time Taken:      {gen_duration:.2f}s")
            print(f"Speed:           {tps:.2f} tokens/sec")
            print(f"------------------")
            
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
