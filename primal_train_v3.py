# ==============================================================================
# PROJECT PRIMAL: PHASE 26 - "SUPERCHARGER" (Async Pipeline)
# ==============================================================================
# Optimization Target: Fix CPU Bottleneck & VRAM "Kill Zone"
# Hardware: Ryzen 5700G (Use 2 Workers) + GTX 1080 Ti
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from torch.utils.data import IterableDataset, DataLoader
import torch.utils.checkpoint
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

# Enable CuDNN Autotuner for Pascal
torch.backends.cudnn.benchmark = True

# ------------------------------------------------------------------------------
# 1. JIT CUDA KERNEL (Phase 25 Engine)
# ------------------------------------------------------------------------------
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__constant__ float c_LUT[16] = {
    0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.333333f, -0.333333f, 
    0.2f, -0.2f, 0.142857f, -0.142857f, 0.090909f, -0.090909f, 
    0.076923f, -0.076923f, 0.0f
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
            if (diff < best_diff) { best_diff = diff; best_idx = i; }
        }
        output[idx] = c_LUT[best_idx] * scale;
    }
}

torch::Tensor quantize_cuda(torch::Tensor input, float scale) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 512; 
    const int blocks = (size + threads - 1) / threads;
    quantize_primal_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), scale, size);
    return output;
}
"""

cpp_source = "torch::Tensor quantize_cuda(torch::Tensor input, float scale);"

print("[*] JIT Compiling Primal Engine (Bypass Active)...")
primal_cuda = load_inline(
    name='primal_cuda_jit_v26',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['quantize_cuda'],
    verbose=False,
    extra_cflags=['-O3', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'],
    extra_cuda_cflags=['--allow-unsupported-compiler', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH']
)

# ------------------------------------------------------------------------------
# 2. OPTIMIZED CONFIGURATION
# ------------------------------------------------------------------------------
CONFIG = {
    "dim": 768,             # Standard Base width
    "n_layers": 12,         # Classic depth
    "n_heads": 12,          # 768 / 12 = 64 dim per head
    "vocab_size": 50257,
    "seq_len": 1024,        # Keep the 1k context!
    
    # HYPER-SPEED TRAINING
    "batch_size": 16,       # We can MASSIVELY increase this now (VRAM is plenty)
    "grad_accum": 8,        # Total Batch = 128 (Faster steps, same weight update)
    
    "lr": 6e-4,             # Smaller models can handle higher Learning Rates
    "max_steps": 100000,
    "save_every": 500,
    "device": "cuda"
}

# ------------------------------------------------------------------------------
# 3. HIGH-PERFORMANCE DATA PIPELINE
# ------------------------------------------------------------------------------
class FineWebStream(IterableDataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
    
    def __iter__(self):
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
        
        # Each worker gets its own connection to the stream
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        buffer = []
        for sample in dataset:
            text = sample['text']
            if len(text) < 200: continue 
            
            tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
            buffer.extend(tokens)
            
            while len(buffer) >= self.seq_len + 1:
                yield torch.tensor(buffer[:self.seq_len + 1], dtype=torch.long)
                buffer = buffer[self.seq_len:]

def get_loader():
    ds = FineWebStream(CONFIG['seq_len'])
    return DataLoader(
        ds, 
        batch_size=CONFIG['batch_size'],
        num_workers=2,          # Use 2 Ryzen Cores for fetching
        pin_memory=True,        # Direct DMA to GPU
        prefetch_factor=4,      # Keep 4 batches ready in RAM
        persistent_workers=False # Avoid zombie processes on streaming
    )

# ------------------------------------------------------------------------------
# 4. MODEL & QUANTIZER
# ------------------------------------------------------------------------------
class PrimalQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        return primal_cuda.quantize_cuda(input, scale.item())
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class PrimalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (0.02 / math.sqrt(24)))
        self.scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return F.linear(x, PrimalQuantFunction.apply(self.weight, self.scale))

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
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=mask, need_weights=False, is_causal=False)
        x = x + attn_out
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
        self.gradient_checkpointing = True 

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Enforce requires_grad for the first checkpoint input
        if self.training:
            x.requires_grad_(True)
            
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                # Closure for the block execution
                def run_block(x_in, m_mask):
                    return block(x_in, mask=m_mask)

                x = torch.utils.checkpoint.checkpoint(
                    run_block, 
                    x, 
                    mask,
                    use_reentrant=False
                )
            else:
                x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ------------------------------------------------------------------------------
# 5. TRAINING LOOP
# ------------------------------------------------------------------------------
def train():
    torch.manual_seed(1337)
    print(f"[*] Supercharger Active. Target: {CONFIG['device']}")
    model = PrimalGPT(CONFIG).to(CONFIG['device'])
    
    if HAS_BNB:
        print("[*] Using 8-bit AdamW (bitsandbytes)")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=CONFIG['lr'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    print("[*] VRAM Pre-Load Check:", torch.cuda.memory_allocated()/1e9, "GB")
    
    loader = get_loader()
    model.train()
    
    step = 0
    accum_loss = 0.0
    t0 = time.time()
    
    optimizer.zero_grad()
    
    print("[*] Filling Prefetch Buffer (This takes ~15s)...")
    
    for i, batch in enumerate(loader):
        # Move to GPU (Non-blocking thanks to pin_memory)
        input_ids = batch[:, :-1].to(CONFIG['device'], non_blocking=True)
        targets = batch[:, 1:].to(CONFIG['device'], non_blocking=True)
        
        # Forward & Backward
        _, loss = model(input_ids, targets=targets)
        loss = loss / CONFIG['grad_accum']
        loss.backward()
        accum_loss += loss.item()
        
        # Optimizer Step
        if (i + 1) % CONFIG['grad_accum'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            
            # Telemetry
            if step % 1 == 0: 
                dt = time.time() - t0
                t0 = time.time()
                # Tokens = Batch * Seq * Accum
                tokens_processed = CONFIG['batch_size'] * CONFIG['seq_len'] * CONFIG['grad_accum']
                tps = tokens_processed / dt
                print(f"Step {step} | Loss: {accum_loss:.4f} | TPS: {tps:.2f} | VRAM: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            
            accum_loss = 0.0
            
            if step >= CONFIG['max_steps']: break
            if step % CONFIG['save_every'] == 0:
                print(f"[*] Saving checkpoint: step {step}")
                torch.save(model.state_dict(), f"primal_0.3B_supercharger_step{step}.pt")

if __name__ == "__main__":
    train()
