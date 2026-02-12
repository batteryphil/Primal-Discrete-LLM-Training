# ==============================================================================
# PROJECT PRIMAL: PHASE 29 - "POLTERGEIST" (REFINED GHOST 0.1B)
# ==============================================================================
# Experimental: Direct Discrete Optimization on 1080 Ti
# FIXES APPLIED: Decoupled Flipping + Adaptive Probability + 2/3 Harmonic Bridge
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import math
import time
import os
import sys

# Forces the terminal output to handle whatever weirdness your model generates
sys.stdout.reconfigure(encoding='utf-8')

# ------------------------------------------------------------------------------
# 1. THE "GHOST" CONFIG
# ------------------------------------------------------------------------------
CONFIG = {
    "dim": 768,
    "n_layers": 13,        # Expanded to 13 layers for Fine-Ghost Refinement
    "n_heads": 12,
    "vocab_size": 50257,
    "seq_len": 512,        # Shortened context for ultra-high TPS
    "batch_size": 16,      # Mega-Batch remains 16
    "grad_accum": 4,       # [CHANGED] 2 -> 4. Effective Batch 64. Reduces noise by 50%.
    'lr': 4.5e-4,          # 1.5x Manual Bump for Phase 38 Breakthrough
    "device": "cuda"
}

# ------------------------------------------------------------------------------
# 2. THE GRID (Your 13-Value IP)
# ------------------------------------------------------------------------------
# Original values: Prime Reciprocals + 0, 1
LUT_VALS = [
    -1.0, -0.5, -0.333333, -0.2, -0.142857, -0.090909, -0.076923, 
    0.0, 
    0.076923, 0.090909, 0.142857, 0.2, 0.333333, 0.5, 1.0
]
PRIMAL_LUT = torch.tensor(LUT_VALS, device=CONFIG['device'])

# Phase 39: The "Fine" Grid (Now with 2/3 Bridge)
# Adds +/- 1/4, 1/6, 1/8, 1/12 AND +/- 2/3 (0.666)
FINE_LUT_VALS = sorted(LUT_VALS + [
    -0.666667, 0.666667, # <--- THE BRIDGE (2/3)
    -0.25, 0.25, 
    -0.166667, 0.166667, 
    -0.125, 0.125, 
    -0.083333, 0.083333
])
FINE_LUT = torch.tensor(FINE_LUT_VALS, device=CONFIG['device'])

# ------------------------------------------------------------------------------
# 3. THE POLTERGEIST QUANTIZER (Decoupled Flipping)
# ------------------------------------------------------------------------------
class GhostQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, scale, lut, vote_buffer, training_mode=True):
        # Phase 29: VRAM Opt - Materialize weights on demand
        # We do NOT pass a pre-allocated weight buffer here. We create it, use it, dropping it.
        if training_mode:
             # In training, we just use the current indices.
             # Jitter is now handled via the "Adaptive Probability" in the step(), not here.
             weights = lut[indices.long()]
        else:
             weights = lut[indices.long()]
            
        weights = weights * scale
        
        # Save context for backward
        ctx.save_for_backward(indices, scale, lut, vote_buffer)
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        indices, scale, lut, vote_buffer = ctx.saved_tensors
        
        # 1. Standard scale gradient
        # Re-materialize weights for gradient calculation (cheap on 1080 Ti)
        weights_proxy = lut[indices.long()]
        grad_scale = (grad_output * weights_proxy).sum()
        
        # 2. Decoupled Flipping (The "Sticky Note" Fix)
        # We do NOT update indices here. We calculate the *desired* direction.
        
        # A. Calculate Direction
        # If scale is negative, a positive gradient means we should flip DOWN the LUT.
        direction = -torch.sign(grad_output * scale.sign()).to(torch.int8)
        
        # B. Z-Score Thresholding (The Strict Teacher)
        grad_abs = torch.abs(grad_output)
        g_mean = grad_abs.mean()
        g_std = grad_abs.std()
        threshold = g_mean + (1.0 * g_std) # 1.0 Sigma
        
        # C. The Vote
        # If signal is strong (above threshold), we cast a vote.
        # Vote = Direction (-1 or +1) * Mask (1 or 0)
        significant_mask = (grad_abs > threshold).to(torch.int8)
        votes = direction * significant_mask
        
        # D. Accumulate Votes
        # We add these votes to the buffer.
        # This handles Gradient Accumulation perfectly: 
        # Batch 1 says +1, Batch 2 says +1 -> Total +2 (Strong vote)
        # Batch 1 says +1, Batch 2 says -1 -> Total 0 (Conflict/Noise)
        vote_buffer.add_(votes)
        
        return None, grad_scale, None, None, None

class GhostLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 1. Indices (Int8/UInt8) - The "DNA"
        raw_w = torch.randn(out_features, in_features) * 0.02
        self.register_buffer('grid_indices', self.quantize_to_indices(raw_w))
        
        # 2. Vote Buffer (Accumulator) - The "Ballot Box"
        # Stores cumulative votes from micro-batches
        self.register_buffer('vote_buffer', torch.zeros(out_features, in_features, dtype=torch.int8))
        
        # 3. Scale (FP32) - The "Volume"
        self.scale = nn.Parameter(torch.tensor(1.0))

    def quantize_to_indices(self, weights):
        w_gpu = weights.to(CONFIG['device'])
        diff = torch.abs(w_gpu.unsqueeze(-1) - PRIMAL_LUT)
        indices = torch.argmin(diff, dim=-1).to(torch.uint8)
        return indices.cpu() 

    def forward(self, x):
        # We pass the vote_buffer to the Function so it can check it out/save it
        weights = GhostQuantFunction.apply(
            self.grid_indices, 
            self.scale, 
            PRIMAL_LUT, 
            self.vote_buffer,
            self.training
        )
        return F.linear(x, weights)
        
    def apply_votes(self, adaptive_prob):
        # CALLED BY OPTIMIZER STEP
        if self.vote_buffer.abs().max() == 0:
            return # No votes cast
            
        # 1. Determine Consensus
        # If accumulation was 2 steps:
        # +2 = Strong Up, -2 = Strong Down, 0 = Conflict, +1/-1 = Weak
        # We can enforce a "Supermajority" if we want, but for now, simple majority.
        final_direction = torch.sign(self.vote_buffer).to(torch.int8)
        
        # 2. Stochastic Application (Adaptive Probability)
        # We mask the updates based on the adaptive probability
        flip_mask = (torch.rand_like(self.vote_buffer.float()) < adaptive_prob).to(torch.int8)
        
        # 3. Apply Update
        update = final_direction * flip_mask
        new_indices = self.grid_indices.int() + update.int()
        
        # 4. Clamp & Commit
        self.grid_indices.copy_(new_indices.clamp(0, len(PRIMAL_LUT) - 1).to(torch.uint8))
        
        # 5. Clear Ballot Box
        self.vote_buffer.zero_()

class FineGhostLinear(GhostLinear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        # Re-register indices using the FINE_LUT
        raw_w = torch.randn(out_features, in_features) * 0.02
        self.register_buffer('grid_indices', self.quantize_to_indices_fine(raw_w))

    def quantize_to_indices_fine(self, weights):
        w_gpu = weights.to(CONFIG['device'])
        diff = torch.abs(w_gpu.unsqueeze(-1) - FINE_LUT)
        indices = torch.argmin(diff, dim=-1).to(torch.uint8)
        return indices.cpu()

    def forward(self, x):
        weights = GhostQuantFunction.apply(
            self.grid_indices, 
            self.scale, 
            FINE_LUT, 
            self.vote_buffer,
            self.training
        )
        return F.linear(x, weights)

# ------------------------------------------------------------------------------
# 4. ARCHITECTURE (Phase 29 "POLTERGEIST" 0.1B)
# ------------------------------------------------------------------------------
class GhostBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c['dim'])
        self.attn = nn.MultiheadAttention(c['dim'], c['n_heads'], batch_first=True)
        self.ln2 = nn.LayerNorm(c['dim'])
        use_fine = c.get('is_fine', False)
        LinearClass = FineGhostLinear if use_fine else GhostLinear

        self.mlp_fc1 = LinearClass(c['dim'], 4 * c['dim'])
        self.mlp_act = nn.GELU(approximate='tanh')
        self.mlp_fc2 = LinearClass(4 * c['dim'], c['dim'])

    def forward(self, x, mask=None):
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=mask, need_weights=False)
        x = x + attn_out
        
        h = self.ln2(x)
        h = self.mlp_fc1(h)
        h = self.mlp_act(h)
        h = self.mlp_fc2(h)
        x = x + h
        return x

class GhostGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.token_emb = nn.Embedding(c['vocab_size'], c['dim'])
        self.pos_emb = nn.Embedding(c['seq_len'], c['dim'])
        layers = []
        for i in range(c['n_layers']):
            layer_cfg = c.copy()
            if i == c['n_layers'] - 1:
                layer_cfg['is_fine'] = True
            layers.append(GhostBlock(layer_cfg))
        self.blocks = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(c['dim'])
        self.head = GhostLinear(c['dim'], c['vocab_size'])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        
        for block in self.blocks:
            if self.training:
                # Poltergeist needs no special args passed through checkpoint
                x = torch.utils.checkpoint.checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= CONFIG['seq_len'] else idx[:, -CONFIG['seq_len']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ------------------------------------------------------------------------------
# 5. DATA PIPELINE (Greedy Buffer Patch)
# ------------------------------------------------------------------------------
class FineWebStream(IterableDataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
    def __iter__(self):
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        token_buffer = []
        for sample in dataset:
            tokens = tokenizer.encode(sample['text']) + [tokenizer.eos_token_id]
            token_buffer.extend(tokens)
            if len(token_buffer) > self.seq_len * 100: 
                while len(token_buffer) >= self.seq_len + 1:
                    yield torch.tensor(token_buffer[:self.seq_len + 1], dtype=torch.long)
                    token_buffer = token_buffer[self.seq_len:]

def get_loader():
    ds = FineWebStream(CONFIG['seq_len'])
    return DataLoader(ds, batch_size=CONFIG['batch_size'], num_workers=4, pin_memory=True)

# ------------------------------------------------------------------------------
# 6. TRAINING LOOP (Poltergeist Edition)
# ------------------------------------------------------------------------------
def train():
    print("[*] Launching 0.1B POLTERGEIST... Decoupled Flipping Active.")
    model = GhostGPT(CONFIG).cuda()
    
    if os.path.exists("primal_ghost_live.pt"):
        print("[*] Resuming from Live Checkpoint...")
        state_dict = torch.load("primal_ghost_live.pt")
        # Removing vote_buffer from load/save logic to avoid mismatch if batch size changed?
        # Actually, vote buffers are registered buffers, so they will load.
        # But we can clear them on resume to be safe.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[*] Loaded Checkpoint. Missing Keys: {len(missing)}")
    
    params_to_opt = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_opt, lr=CONFIG['lr'])
    
    loader = get_loader()
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model.train()
    
    t0 = time.time()
    for i, batch in enumerate(loader):
        input_ids = batch[:, :-1].cuda(non_blocking=True)
        targets = batch[:, 1:].cuda(non_blocking=True)
        
        logits, loss = model(input_ids, targets=targets)
        loss = loss / CONFIG['grad_accum']
        loss.backward()
        
        # OPTIMIZER STEP (The Poltergeist Update)
        if (i + 1) % CONFIG['grad_accum'] == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # --- APPLY DISCRETE VOTES ---
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, GhostLinear):
                        # Adaptive Flip Probability
                        # If Scale is near 1.0 (Confident), Prob -> 0.1%
                        # If Scale < 0.5 (Unsure), Prob -> 2.0%
                        # We use 0.002 as base, multiply by (1/Scale)
                        # Clamped to [0.0001, 0.05]
                        scale_val = abs(m.scale.item())
                        adaptive_prob = 0.002 / (scale_val + 1e-6)
                        adaptive_prob = max(0.0001, min(0.05, adaptive_prob))
                        
                        m.apply_votes(adaptive_prob)
            
            dt = time.time() - t0
            t0 = time.time()
            tps = (CONFIG['batch_size'] * CONFIG['seq_len'] * CONFIG['grad_accum']) / dt
            
            step = i//CONFIG['grad_accum'] + 1
            
            # Monitoring
            primal_scales = [m.scale for m in model.modules() if isinstance(m, GhostLinear) and not isinstance(m, FineGhostLinear)]
            fine_scales = [m.scale for m in model.modules() if isinstance(m, FineGhostLinear)]
            p_scale = torch.stack(primal_scales).mean().item() if primal_scales else 0.0
            f_scale = torch.stack(fine_scales).mean().item() if fine_scales else 0.0
            
            print(f"Step {step} | Loss: {loss.item()*CONFIG['grad_accum']:.4f} | TPS: {tps:.2f} | P-Scale: {p_scale:.4f} | F-Scale: {f_scale:.4f} | VRAM: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            
            if step % 100 == 0:
                torch.save(model.state_dict(), "primal_ghost_live.pt")
                print(f"[*] Saved Live Checkpoint at Step {step}")
            
            if step % 50 == 0:
                print(f"\n[*] RUNNING STEP {step} SALAD TEST...")
                model.eval()
                with torch.no_grad():
                    test_prompt = "The future of AI is"
                    tokens = tokenizer.encode(test_prompt, return_tensors="pt").cuda()
                    gen_tokens = model.generate(tokens, max_new_tokens=20)
                    output_text = tokenizer.decode(gen_tokens[0])
                    safe_output = output_text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
                    print(f"--- STEP {step} SALAD TEST: {safe_output} ---\n")
                model.train()

if __name__ == "__main__":
    train()
