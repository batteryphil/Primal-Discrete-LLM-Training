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
        self.register_buffer('grid', get_prime_grid())
    def forward(self, x):
        w_quant = PrimeQuantFunction.apply(self.weight, self.grid)
        return F.linear(x, w_quant, self.bias)

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
    args = parser.parse_args()

    print("--- PROJECT TRINITY: INFERENCE ENGINE (V1.0.0) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Loading Base Architecture...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    
    # Inject Prime Architecture
    replacements = []
    for name, module in model.named_modules():
         if isinstance(module, nn.Linear) and "head" not in name: replacements.append((name, module))
    for name, module in replacements:
        parent = model.get_submodule(name.rsplit('.', 1)[0] if '.' in name else '')
        setattr(parent, name.rsplit('.', 1)[-1], PrimeLinear(module).to(DEVICE))

    # Load and Unpack Binary
    load_trinity_bin(model, args.model)
    model.eval()

    print(f"\nPrompt: {args.prompt}")
    tokens = tokenizer(args.prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(
            **tokens, 
            max_new_tokens=32, 
            temperature=0.6, 
            repetition_penalty=1.5,
            do_sample=True
        )
    
    print(f"\nResult: {tokenizer.decode(output[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
