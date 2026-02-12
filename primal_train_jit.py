# ==============================================================================
# PROJECT PRIMAL: PHASE 25 - "NITROUS" (CUDA JIT EDITION)
# ==============================================================================
# Hardware: Ryzen 5700G + GTX 1080 Ti (11GB VRAM)
# Strategy: JIT-Compiled CUDA Kernel for 4-bit Quantization
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math
import time
import os
import sys

# Optional bitandbytes check
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# ------------------------------------------------------------------------------
# 1. THE C++ CUDA KERNEL (Written as a string, compiled on run)
# ------------------------------------------------------------------------------
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Hardcoded Primal LUT (15 values + 0 pad)
__constant__ float c_LUT[16] = {
    0.0f, 
    1.0f,       -1.0f, 
    0.5f,       -0.5f, 
    0.333333f,  -0.333333f, 
    0.2f,       -0.2f, 
    0.142857f,  -0.142857f, 
    0.090909f,  -0.090909f, 
    0.076923f,  -0.076923f,
    0.0f
};

__global__ void quantize_primal_kernel(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       float scale, 
                                       int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] / scale;
        
        float best_diff = 1e9f;
        int best_idx = 0;
        
        #pragma unroll
        for (int i = 0; i < 15; ++i) {
            float diff = fabsf(val - c_LUT[i]);
            if (diff < best_diff) {
                best_diff = diff;
                best_idx = i;
            }
        }
        
        output[idx] = c_LUT[best_idx] * scale;
    }
}

torch::Tensor quantize_cuda(torch::Tensor input, float scale) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int threads = 512; // Sweet spot for Pascal
    const int blocks = (size + threads - 1) / threads;
    
    quantize_primal_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale,
        size
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor quantize_cuda(torch::Tensor input, float scale);
"""

# JIT Compile the Kernel
print("[*] Compiling Primal CUDA Kernel (Nitrous Injection)...")
try:
    primal_cuda = load_inline(
        name='primal_cuda_jit',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['quantize_cuda'],
        verbose=True,
        extra_cflags=['-O3', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'],
        extra_cuda_cflags=['--allow-unsupported-compiler', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH']
    )
    print("[*] Compilation Complete. Primal Engine is Native.")
except Exception as e:
    print(f"[!] CUDA JIT Compilation Failed: {e}")
    print("[!] Ensure Visual Studio Build Tools and NVCC are in PATH.")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. CONFIGURATION (Still tuned for 0.3B on 1080 Ti)
# ------------------------------------------------------------------------------
CONFIG = {
    "vocab_size": 50257,
    "dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "seq_len": 1024,
    "batch_size": 1,      # Lower micro-batch for stability on 11GB
    "grad_accum": 128,    # Effective Batch = 128
    "lr": 2e-4,
    "max_steps": 100000,
    "save_every": 1000,
    "device": "cuda"
}

# ------------------------------------------------------------------------------
# 3. THE ACCELERATED QUANTIZER
# ------------------------------------------------------------------------------
class PrimalQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        # Calls the C++ Kernel directly!
        return primal_cuda.quantize_cuda(input, scale.item())

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Pass-through
        return grad_output, None

class PrimalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Init: Scaled for depth
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (0.02 / math.sqrt(24)))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        w_quant = PrimalQuantFunction.apply(self.weight, self.scale)
        return F.linear(x, w_quant)

# ------------------------------------------------------------------------------
# 4. MODEL (Standard Primal GPT)
# ------------------------------------------------------------------------------
class PrimalBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c['dim'])
        self.attn = nn.MultiheadAttention(c['dim'], c['n_heads'], batch_first=True)
        self.ln2 = nn.LayerNorm(c['dim'])
        self.mlp = nn.Sequential(
            PrimalLinear(c['dim'], 4 * c['dim']),
            nn.GELU(approximate='tanh'),
            PrimalLinear(4 * c['dim'], c['dim'])
        )

    def forward(self, x, mask=None):
        # Standard Attention
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=mask, need_weights=False, is_causal=False)
        x = x + attn_out
        # MLP
        x = x + self.mlp(self.ln2(x))
        return x

class PrimalGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.token_emb = nn.Embedding(c['vocab_size'], c['dim'])
        self.pos_emb = nn.Embedding(c['seq_len'], c['dim'])
        self.blocks = nn.ModuleList([PrimalBlock(c) for _ in range(c['n_layers'])])
        self.ln_f = nn.LayerNorm(c['dim'])
        self.head = PrimalLinear(c['dim'], c['vocab_size'])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Causal Mask (Manual)
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ------------------------------------------------------------------------------
# 5. STREAMING DATA & LOOP
# ------------------------------------------------------------------------------
def get_streaming_loader():
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    print("[*] Connecting to HuggingFace Stream (FineWeb-Edu)...")
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"[!] Dataset Loading Failed: {e}")
        sys.exit(1)
    
    def data_generator():
        buffer = []
        for sample in dataset:
            text = sample['text']
            if len(text) < 100: continue
            tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
            buffer.extend(tokens)
            while len(buffer) >= CONFIG['seq_len'] + 1:
                chunk = buffer[:CONFIG['seq_len'] + 1]
                buffer = buffer[CONFIG['seq_len']:]
                yield torch.tensor(chunk)
                
    return data_generator()

def train():
    torch.manual_seed(1337)
    
    # Init Model
    print("[*] Initializing 0.3B Model...")
    model = PrimalGPT(CONFIG).to(CONFIG['device'])
    
    # Optimizer
    if HAS_BNB:
        print("[*] Using 8-bit AdamW (bitsandbytes)")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=CONFIG['lr'])
    else:
        print("[!] bitsandbytes not found, falling back to torch.optim.AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    # Data
    loader = get_streaming_loader()
    iter_stream = iter(loader)
    
    print("[*] Training Started (NITROUS Accelerated)...")
    model.train()
    optimizer.zero_grad()
    
    step = 0
    total_loss = 0
    tokens_in_window = 0
    t0 = time.time()
    
    while step < CONFIG['max_steps']:
        step_loss = 0
        for _ in range(CONFIG['grad_accum']):
            try:
                data = next(iter_stream).to(CONFIG['device'])
            except StopIteration:
                print("[*] Data stream exhausted.")
                return

            input_ids = data[:-1].unsqueeze(0)
            targets = data[1:].unsqueeze(0)
            
            logits, loss = model(input_ids, targets=targets)
            loss = loss / CONFIG['grad_accum']
            loss.backward()
            step_loss += loss.item()
            tokens_in_window += CONFIG['seq_len']

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        
        if step % 1 == 0:
            dt = time.time() - t0
            tps = (tokens_in_window) / dt
            print(f"Step {step} | Loss: {step_loss:.4f} | TPS: {tps:.2f} | VRAM: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            t0 = time.time()
            tokens_in_window = 0
            
        if step % CONFIG['save_every'] == 0:
            print(f"[*] Saving: primal_0.3B_jit_step{step}.pt")
            torch.save(model.state_dict(), f"primal_0.3B_jit_step{step}.pt")

if __name__ == "__main__":
    train()
