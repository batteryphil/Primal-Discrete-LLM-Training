# ==============================================================================
# PROJECT PRIMAL: PHASE 24 - "ANTIGRAVITY" 0.3B BUILD
# ==============================================================================
# Hardware: Ryzen 5700G + GTX 1080 Ti (11GB VRAM)
# Strategy: Streaming Data (No SSD Hit) + 8-bit Optim + Primal QAT
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2TokenizerFast, AutoTokenizer
from datasets import load_dataset
# import bitsandbytes as bnb # Optional if 8-bit optim needed
import math
import time
import os
import sys

# ------------------------------------------------------------------------------
# 1. CONFIGURATION (The 0.3B Sweet Spot)
# ------------------------------------------------------------------------------
CONFIG = {
    # Model Specs (Approx GPT-2 Medium)
    "vocab_size": 50257,
    "dim": 1024,            # Width
    "n_layers": 24,         # Depth (Reasoning capability)
    "n_heads": 16,          # 1024/16 = 64 head dim
    "seq_len": 1024,        # Context Window
    
    # Training (Pascal Safe - Optimized for 11GB)
    "batch_size": 1,        # Ultra-micro batch
    "grad_accum": 128,      # Effective Batch = 128
    "lr": 2e-4,             # slightly lower for stability
    "max_steps": 100000,
    "save_every": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"[*] Booting Antigravity Engine on {CONFIG['device']} ({torch.cuda.get_device_name(0) if CONFIG['device']=='cuda' else 'CPU'})")

# ------------------------------------------------------------------------------
# 2. THE PRIMAL 4-BIT QUANTIZER (YOUR IP)
# ------------------------------------------------------------------------------
# The 13-Value Harmonic Grid (Padded to 16 for alignment)
PRIMAL_LUT = torch.tensor([
    0.0, 
    1.0,       -1.0, 
    0.5,       -0.5, 
    0.333333,  -0.333333, 
    0.2,       -0.2, 
    0.142857,  -0.142857, 
    0.090909,  -0.090909, 
    0.076923,  -0.076923,
    0.0 # Pad
], device=CONFIG['device'])

class PrimalQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale):
        # 1. Normalize
        w_scaled = weight / scale
        
        # 2. Snap to nearest Prime Value (Chunked to save VRAM)
        # Broadcasting: (N, 1) - (1, 16) -> (N, 16)
        # Head layer: 50M params -> 800M elements -> 3.2GB VRAM if done at once
        
        w_flat = w_scaled.view(-1)
        n_elements = w_flat.numel()
        chunk_size = 1024 * 1024 * 4 # 4M elements per chunk (~256MB expansion)
        w_quant_flat = torch.zeros_like(w_flat)
        
        lut_expanded = PRIMAL_LUT.view(1, -1)
        
        for i in range(0, n_elements, chunk_size):
            end = min(i + chunk_size, n_elements)
            chunk = w_flat[i:end]
            
            # Expand and find closest
            chunk_expanded = chunk.unsqueeze(-1)
            diff = torch.abs(chunk_expanded - lut_expanded)
            best_indices = torch.argmin(diff, dim=-1)
            
            w_quant_flat[i:end] = PRIMAL_LUT[best_indices]
            
        # 3. Decode
        return w_quant_flat.view_as(weight) * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE)
        # Allows gradients to flow through the discrete grid
        return grad_output, None

class PrimalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize Latent Weights (FP32)
        # Scaled init for depth (1/sqrt(24)) to prevent exploding grads
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (0.02 / math.sqrt(24)))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        w_quant = PrimalQuantizer.apply(self.weight, self.scale)
        return F.linear(x, w_quant)

# ------------------------------------------------------------------------------
# 3. MODEL ARCHITECTURE (0.3B)
# ------------------------------------------------------------------------------
class PrimalBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c['dim'])
        self.attn = nn.MultiheadAttention(c['dim'], c['n_heads'], batch_first=True)
        self.ln2 = nn.LayerNorm(c['dim'])
        # MLP is where the Parameters live -> Use PrimalLinear here!
        self.mlp = nn.Sequential(
            PrimalLinear(c['dim'], 4 * c['dim']),
            nn.GELU(approximate='tanh'), # GPT-2 style
            PrimalLinear(4 * c['dim'], c['dim'])
        )

    def forward(self, x):
        B, T, C = x.shape
        # Create Causal Mask
        # mask[i, j] = -inf if j > i else 0
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        
        # Standard Attention
        # Note: attention mask shape (T, T) is supported
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), 
                                attn_mask=mask,
                                need_weights=False, is_causal=False) # Manually passing mask
        x = x + attn_out
        # Quantized MLP
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
        
        # Add position embeddings
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Forward through blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        
        if targets is not None:
            # Calculate logits and loss
            logits = self.head(x)
            # Flatten for CE loss
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
            return logits, loss
        else:
            # Inference only: usually only last token needed
            logits = self.head(x[:, [-1], :]) 
            return logits, None

# ------------------------------------------------------------------------------
# 4. DATA PIPELINE (Streaming TinyStories with Packing)
# ------------------------------------------------------------------------------
class TokenBuffer:
    def __init__(self, tokenizer, batch_size, seq_len):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.buffer = []
        self.eos_token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256 # Default GPT-2 EOS

    def get_batch(self, data_iter):
        inputs = []
        targets = []
        
        while len(inputs) < self.batch_size:
            # Fill buffer
            while len(self.buffer) < self.seq_len + 1:
                try:
                    sample = next(data_iter)
                    text = sample['text']
                    tokens = self.tokenizer.encode(text) + [self.eos_token]
                    self.buffer.extend(tokens)
                except StopIteration:
                    if len(self.buffer) == 0:
                        return None, None # Truly done
                    break # Use what we have

            if len(self.buffer) >= self.seq_len + 1:
                chunk = self.buffer[:self.seq_len + 1]
                self.buffer = self.buffer[self.seq_len + 1:]
                
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                inputs.append(x)
                targets.append(y)
            else:
                # Not enough for a full sequence, discard remainder? 
                # Or pad? Let's discard for simplicity in streaming training.
                break

        if len(inputs) == 0:
            return None, None

        x_batch = torch.stack(inputs)
        y_batch = torch.stack(targets)
        return x_batch, y_batch


# ------------------------------------------------------------------------------
# 5. TRAINING LOOP
# ------------------------------------------------------------------------------
def main():
    torch.manual_seed(1337)
    
    # 1. Setup Data
    print("[*] Loading Tokenizer (GPT2)...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    print("[*] Streaming Dataset (TinyStories)...")
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        data_iter = iter(dataset)
    except Exception as e:
        print(f"[!] Dataset Error: {e}")
        return

    pipeline = TokenBuffer(tokenizer, CONFIG['batch_size'], CONFIG['seq_len'])

    # 2. Setup Model
    print("[*] Initializing Model...")
    model = PrimalGPT(CONFIG).to(CONFIG['device'])
    
    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model Parameters: {n_params/1e6:.2f}M (0.3B Class)")
    
    # 3. Optimizer
    # Using 8-bit AdamW if available to save VRAM, else standard AdamW
    try:
        import bitsandbytes as bnb
        print("[*] Using 8-bit AdamW (bitsandbytes)")
        # Make sure bitsandbytes is actually working
        if not bnb.cuda_setup.main.is_available():
            raise ImportError("Bitsandbytes CUDA setup failed")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=CONFIG['lr'])
    except Exception as e:
        print(f"[!] bitsandbytes not found/failed ({e}), falling back to torch.optim.AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])

    # 4. Loop
    print(f"[*] Starting Training Loop (Max Steps: {CONFIG['max_steps']})")
    model.train()
    
    start_time = time.time()
    accum_loss = 0.0
    
    for step in range(CONFIG['max_steps']):
        
        # Gradient Accumulation Loop
        step_loss = 0.0
        
        for micro_step in range(CONFIG['grad_accum']):
            x, y = pipeline.get_batch(data_iter)
            
            if x is None:
                print("Data stream ended.")
                return
            
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])

            # Forward
            logits, loss = model(x, y)
            loss = loss / CONFIG['grad_accum'] # Scale loss
            step_loss += loss.item()
            
            # Backward
            loss.backward()
            
        # Step
        # Clip grads
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        accum_loss = step_loss # Current step loss
        
        # Telemetry
        if step % 10 == 0:
            dt = time.time() - start_time
            tokens_processed = CONFIG['batch_size'] * CONFIG['grad_accum'] * CONFIG['seq_len'] * 10 
            # Note: This is processed per 10 steps
            if step == 0: tokens_processed /= 10 # First step edge case
            
            tps = tokens_processed / dt if dt > 0 else 0
            start_time = time.time()
            
            mem = torch.cuda.memory_reserved()/1e9 if torch.cuda.is_available() else 0
            print(f"Step {step} | Loss: {accum_loss:.4f} | TPS: {tps:.2f} | GPU Mem: {mem:.2f}GB")
        
        if step > 0 and step % CONFIG['save_every'] == 0:
            print(f"[*] Saving checkpoint: primal_0.3b_step{step}.pt")
            torch.save(model.state_dict(), f"primal_0.3b_step{step}.pt")

if __name__ == "__main__":
    main()
