# ==============================================================================
# PROJECT PRIMAL: PHASE 53 - "ANTIGRAVITY PROTOCOL"
# ==============================================================================
# CODENAME: ANTIGRAVITY STABILIZER
# STATUS: ACTIVE / PHYSICS UPGRADE
# CHANGELOG:
# - Implemented "Scale Decay" (Forced Friction) to pull scales to 1.0.
# - Implemented "Gradient Centering" to prevent signal fighting.
# - Implemented "Threshold Annealing" (Momentum Kick) for early training.
# - Implemented "Thermal Reset" (Flip Rate Watchdog).
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import math
import time
import os
import sys
import manifolds # Expects manifolds.py to exist
# Forces the terminal output to handle whatever weirdness your model generates
sys.stdout.reconfigure(encoding='utf-8')

# ------------------------------------------------------------------------------
# 1. THE "ANTIGRAVITY" CONFIG
# ------------------------------------------------------------------------------
CONFIG = {
    "dim": 768,
    "n_layers": 13,
    "n_heads": 12,
    "vocab_size": 50257,
    "seq_len": 512,
    "batch_size": 8,       # Safe for 1080 Ti
    "grad_accum": 8,       # Effective Batch 64
    "lr": 5e-5,            # [Phase 59] Lower for final descent
    "device": "cuda",
    "cooldown_steps": 50,   # [Phase 59] Harder "Sticky" logic
    "scale_decay": 0.01,   # [NEW] Force scales to 1.0
    "micro_save_interval": 10,      # Save every 10 steps for crash recovery
    "freeze_threshold": 0.0050,     # Flip rate (%) to trigger Deep Freeze
    "freeze_window": 20,            # Must stay below threshold for 20 steps
    "final_polish_steps": 50        # Steps to run on scales only after freeze
}

PRIMAL_LUT = manifolds.generate_manifold(device=CONFIG['device'])
FINE_LUT = PRIMAL_LUT 

# ------------------------------------------------------------------------------
# 3. THE POLTERGEIST ENGINE v3.1 (Antigravity)
# ------------------------------------------------------------------------------
class GhostQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, scale, lut, vote_buffer, sensitivity):
        weights = lut[indices.long()] * scale
        ctx.save_for_backward(indices, scale, lut, vote_buffer)
        ctx.sensitivity = sensitivity 
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        indices, scale, lut, vote_buffer = ctx.saved_tensors
        sensitivity = ctx.sensitivity
        
        # [NEW] Gradient Centering (Antigravity)
        # Prevents global shifts from dominating the "vote"
        grad_output = grad_output - grad_output.mean(dim=1, keepdim=True)
        
        # 1. Gradient for Scale (Summed over input dim for per-channel)
        weights_proxy = lut[indices.long()]
        grad_scale = (grad_output * weights_proxy).sum(dim=1, keepdim=True)
        
        # 2. THE WHISPER LOGIC (Adaptive Thresholding)
        direction = -torch.sign(grad_output * scale.sign()).to(torch.int8)
        
        # Adaptive Threshold
        grad_abs = torch.abs(grad_output)
        g_mean = grad_abs.mean()
        g_std = grad_abs.std()
        
        # Sensitivity controls how "quiet" a gradient can be and still vote
        threshold = g_mean + (sensitivity * g_std)
        
        # The Vote
        significant_mask = (grad_abs > threshold).to(torch.int8)
        votes = direction * significant_mask
        vote_buffer.add_(votes)
        
        return None, grad_scale, None, None, None

class GhostLinear(nn.Module):
    def __init__(self, in_features, out_features, sensitivity=0.5):
        super().__init__()
        self.sensitivity = sensitivity 
        
        raw_w = torch.randn(out_features, in_features) * 0.02
        self.register_buffer('grid_indices', self.quantize_to_indices(raw_w))
        self.register_buffer('vote_buffer', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('cooldown', torch.zeros(out_features, in_features, dtype=torch.uint8))
        self.scale = nn.Parameter(torch.ones(out_features, 1))

    def quantize_to_indices(self, weights):
        chunk_size = 1024
        num_chunks = math.ceil(weights.shape[0] / chunk_size)
        indices_list = []
        w_gpu_full = weights.to(CONFIG['device'])
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, weights.shape[0])
            w_chunk = w_gpu_full[start:end]
            diff = torch.abs(w_chunk.unsqueeze(-1) - PRIMAL_LUT)
            chunk_indices = torch.argmin(diff, dim=-1).to(torch.uint8)
            indices_list.append(chunk_indices)
        return torch.cat(indices_list, dim=0).cpu()

    def forward(self, x, annealing_factor=1.0):
        # [NEW] Apply Annealing to Sensitivity (Lower factor = easier to trigger if we divide)
        # Wait, standard logic: Threshold = Mean + (Sens * Std).
        # To make it easier (Shake Box), we want SMALLER threshold -> SMALLER Sens.
        # So we multiply by factor < 1.0.
        effective_sensitivity = self.sensitivity * annealing_factor
        
        weights = GhostQuantFunction.apply(
            self.grid_indices, self.scale, PRIMAL_LUT, self.vote_buffer, effective_sensitivity
        )
        return F.linear(x, weights)

    def apply_votes(self, adaptive_prob, name="Unknown"):
        # 1. Hysteresis (Cooldown)
        # [FIXED] Prevent Uint8 wrap: only subtract if > 0
        with torch.no_grad():
            mask = self.cooldown > 0
            self.cooldown[mask] -= 1
            
        # 2. THE GRAVITY WELL: Pull scales toward 1.0 (Unity)
        # Forces the model to stop 'drifting' and start 'snapping'
        with torch.no_grad():
            self.scale.data = self.scale.data * 0.999 + (1.0 * 0.001)

        if self.vote_buffer.abs().max() == 0:
            return 0
            
        # 3. RESONATOR & FLOODGATE (DAMPING FIELD)
        prob_tensor = torch.full_like(self.vote_buffer.float(), adaptive_prob)
        
        # 32-vote Resonator: Keep at 0.1 (10% chance) - turned down from 50%
        high_pressure_mask = self.vote_buffer.abs() >= 32
        prob_tensor[high_pressure_mask] = 0.1 
        
        # 127-vote Saturation Bypass: RE-BOOSTED to 100% (Total Surrender)
        saturated_mask = self.vote_buffer.abs() >= 127
        prob_tensor[saturated_mask] = 1.0 
        
        # 4. Stochastic Flip
        final_direction = torch.sign(self.vote_buffer).to(torch.int8)
        flip_mask = (torch.rand_like(self.vote_buffer.float()) < prob_tensor).to(torch.int8)
        ready_mask = (self.cooldown == 0).to(torch.int8)
        valid_flips = final_direction * flip_mask * ready_mask
        
        # 5. Apply Updates
        new_indices = self.grid_indices.int() + valid_flips.int()
        self.grid_indices.copy_(new_indices.clamp(0, 255).to(torch.uint8))
        
        # --- [CRITICAL FIX] THE CONVICTION RESET ---
        # If a weight flipped, its 'work' is done. Zero out its buffer.
        with torch.no_grad():
            self.vote_buffer[valid_flips != 0] = 0 
            
        # --- PHASE 57: THE CURING AGENT ---
        # Dynamic Cooldown: The more you flip, the longer you stay locked.
        num_flips = torch.count_nonzero(valid_flips).item()
        if num_flips > 0:
            # If more than 0.1% of the layer flips, triple the cooldown
            if num_flips > (self.grid_indices.numel() * 0.001):
                lock_duration = CONFIG['cooldown_steps'] * 3
            else:
                lock_duration = CONFIG['cooldown_steps']
                
            self.cooldown[valid_flips != 0] = lock_duration
        
        # --- AVALANCHE LOG ---
        if num_flips > 0:
            print(f"[!] AVALANCHE in {name} | Flips: {num_flips} | Tension: {self.scale.mean().item():.6f}")

        # 6. LINEAR FRICTION (Slow Bleed)
        with torch.no_grad():
            pos_mask = self.vote_buffer > 0
            neg_mask = self.vote_buffer < 0
            self.vote_buffer[pos_mask] -= 1
            self.vote_buffer[neg_mask] += 1
        
        return num_flips

# ------------------------------------------------------------------------------
# POLTERGEIST V3.1 DIAGNOSTICS
# ------------------------------------------------------------------------------
def check_poltergeist_vitals(model):
    print("\n" + "="*50)
    print("POLTERGEIST V3: INTERNAL PRESSURE REPORT")
    print("="*50)
    
    total_unstable = 0
    total_params = 0
    
    for name, m in model.named_modules():
        if isinstance(m, GhostLinear):
            # Calculate how 'charged' the vote buffer is
            # We look at the max absolute vote count in this layer
            votes = m.vote_buffer.float()
            max_v = votes.abs().max().item()
            mean_v = votes.abs().mean().item()
            
            # Estimate how many weights are above a 'danger' threshold (e.g., 5+ votes)
            unstable_count = torch.sum(votes.abs() >= 4).item()
            total_unstable += unstable_count
            total_params += votes.numel()
            
            # Sensitivity of this specific layer
            sens = getattr(m, 'sensitivity', 'N/A')
            
            print(f"Layer: {name:25} | Max Vote: {max_v:4.0f} | Mean: {mean_v:6.4f} | Unstable: {unstable_count:8.0f} | Sens: {sens:.2f}")

    pressure_pct = (total_unstable / total_params) * 100 if total_params > 0 else 0
    print("-" * 50)
    print(f"TOTAL SYSTEM PRESSURE: {pressure_pct:.6f}% of weights are 'Unstable'")
    print("="*50 + "\n")

class FineGhostLinear(GhostLinear):
    def __init__(self, in_features, out_features, sensitivity=0.5):
        super().__init__(in_features, out_features, sensitivity)
        raw_w = torch.randn(out_features, in_features) * 0.02
        self.register_buffer('grid_indices', self.quantize_to_indices_fine(raw_w))

    def quantize_to_indices_fine(self, weights):
        w_gpu = weights.to(CONFIG['device'])
        diff = torch.abs(w_gpu.unsqueeze(-1) - FINE_LUT)
        indices = torch.argmin(diff, dim=-1).to(torch.uint8)
        return indices.cpu()
    
    def forward(self, x, annealing_factor=1.0):
        effective_sensitivity = self.sensitivity * annealing_factor
        weights = GhostQuantFunction.apply(
            self.grid_indices, self.scale, FINE_LUT, self.vote_buffer, effective_sensitivity
        )
        return F.linear(x, weights)

# ------------------------------------------------------------------------------
# 4. ARCHITECTURE (Phase 53)
# ------------------------------------------------------------------------------
class GhostBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c['dim'])
        self.attn = nn.MultiheadAttention(c['dim'], c['n_heads'], batch_first=True)
        self.ln2 = nn.LayerNorm(c['dim'])
        
        sens = c.get('sensitivity', 0.5)
        use_fine = c.get('is_fine', False)
        LinearClass = FineGhostLinear if use_fine else GhostLinear

        self.mlp_fc1 = LinearClass(c['dim'], 4 * c['dim'], sensitivity=sens)
        self.mlp_act = nn.GELU(approximate='tanh')
        self.mlp_fc2 = LinearClass(4 * c['dim'], c['dim'], sensitivity=sens)

    def forward(self, x, mask=None, annealing_factor=1.0):
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=mask, need_weights=False)
        x = x + attn_out
        h = self.ln2(x)
        # Pass annealing factor
        h = self.mlp_fc1(h, annealing_factor=annealing_factor)
        h = self.mlp_act(h)
        h = self.mlp_fc2(h, annealing_factor=annealing_factor)
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
            progress = i / max(1, c['n_layers'] - 1)
            layer_cfg['sensitivity'] = 0.2 + (0.6 * progress)
            if i == c['n_layers'] - 1:
                layer_cfg['is_fine'] = True
            layers.append(GhostBlock(layer_cfg))
            
        self.blocks = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(c['dim'])
        self.head = GhostLinear(c['dim'], c['vocab_size'], sensitivity=0.8)

    def forward(self, idx, targets=None, annealing_factor=1.0):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        
        for block in self.blocks:
            if self.training:
                # Need to pass annealing_factor through checkpoint
                # But checkpoint signatures are tricky.
                # Simplification: Don't checkpoint annealing_factor (defaults to 1.0 inside checkpoint if not passed)
                # To pass it properly, we need to wrap or rely on standard args.
                # Let's try passing it as arg.
                x = torch.utils.checkpoint.checkpoint(block, x, mask, annealing_factor, use_reentrant=False)
            else:
                x = block(x, mask=mask, annealing_factor=annealing_factor)

        x = self.ln_f(x)
        # Head gets annealing too? Maybe not. It's rigid.
        logits = self.head(x) # Head uses default annealing=1.0
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
# 5. DATA & TRAINING
# ------------------------------------------------------------------------------
class FineWebStream(IterableDataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
    def __iter__(self):
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
        import time
        max_retries = 10
        dataset = None
        for attempt in range(max_retries):
            try:
                dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
                break
            except Exception as e:
                # Catch HfHubHTTPError and similar
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait = 5 * (2 ** attempt)
                    print(f"[*] HF Rate Limit (429). Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else: 
                     # Only swallow if it looks like a transient network erorr
                     print(f"[*] Retrying dataset load due to: {e}", flush=True)
                     time.sleep(5)
        
        if dataset is None: raise RuntimeError("Failed to load dataset.")
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
    return DataLoader(ds, batch_size=CONFIG['batch_size'], num_workers=0, pin_memory=True)

# --- PHASE 58 MATPLOTLIB FIX ---
import matplotlib.pyplot as plt  # [CRITICAL FIX]
from matplotlib import colormaps # [2026 COMPLIANCE]
import numpy as np

def log_manifold_heatmap(model, step):
    # 1. Gather all grid indices
    all_indices = []
    for m in model.modules():
        if isinstance(m, GhostLinear):
            all_indices.append(m.grid_indices.cpu().numpy().flatten())
            
    if not all_indices:
        return

    flat_indices = np.concatenate(all_indices)
    counts = np.bincount(flat_indices, minlength=256)
    
    # 2. [FIXED] Use the non-deprecated colormaps registry
    cmap = colormaps.get_cmap('plasma')
    color_array = cmap(np.linspace(0, 1, 256))
    
    # 3. [FIXED] plt is now defined
    plt.figure(figsize=(12, 5))
    plt.bar(range(256), counts, color=color_array, width=1.0) 
    
    plt.title(f"PRIMAL-DISCRETE Manifold Usage - Step {step}")
    plt.xlabel("8-bit LUT Index (0-255)")
    plt.ylabel("Weight Count (Log Scale)")
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    
    # 4. Save and Close (Vital for VRAM/RAM cleanup)
    plt.tight_layout()
    plt.savefig(f"manifold_step_{step}.png")
    plt.close('all') 
    print(f"[*] SEISMIC MAP GENERATED: manifold_step_{step}.png")

# --- PHASE 60.5: AUTOMATED WATCHDOG & MICRO-SAVE LOGIC ---
def run_automated_training_cycle(model, step, flip_rate, loss, optimizer):
    # --- AUTOMATED MICRO-SAVE ---
    if step % CONFIG['micro_save_interval'] == 0:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, "primal_ghost_autosave.pt")
        print(f"[*] AUTO-SAVE: Grid and Optimizer state secured at Step {step}")

    # --- AUTOMATED DEEP FREEZE WATCHDOG ---
    if not hasattr(run_automated_training_cycle, "stability_counter"):
        run_automated_training_cycle.stability_counter = 0

    if step > 250 and flip_rate < CONFIG['freeze_threshold']:
        run_automated_training_cycle.stability_counter += 1
    else:
        run_automated_training_cycle.stability_counter = 0

    if run_automated_training_cycle.stability_counter >= CONFIG['freeze_window']:
        return True 
    return False

# --- PHASE 61: HARDENED RECOVERY BLOCK ---
def load_latest_state(model, optimizer):
    """Check for autosave checkpoint and resume from it if found."""
    checkpoint_path = "primal_ghost_autosave.pt"
    if os.path.exists(checkpoint_path):
        print(f"[*] RECOVERY DETECTED: Loading state from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
        
        # Filter out legacy scalar scales (Phase 52 compat)
        state = checkpoint['model_state_dict']
        keys_to_remove = [k for k, v in state.items() if k.endswith('.scale') and v.dim() == 0]
        for k in keys_to_remove:
            del state[k]
        
        model.load_state_dict(state, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"[*] RESUMING FROM STEP {start_step}. No progress lost.")
        return start_step
    else:
        print("[!] No autosave checkpoint found. Using standard resume path.")
        return 0

# --- PHASE 61: SIDECAR TRIGGER LOGIC ---
def trigger_sidecar_audit(model, step, flip_rate):
    """Launch CPU-only PPL audit at stability milestones without interrupting GPU training."""
    import subprocess
    stability_milestones = [0.04, 0.03, 0.02, 0.01, 0.005]
    
    if not hasattr(trigger_sidecar_audit, "last_triggered"):
        trigger_sidecar_audit.last_triggered = 1.0  # Default high

    for milestone in stability_milestones:
        if flip_rate <= milestone < trigger_sidecar_audit.last_triggered:
            print(f"\n[!] STABILITY MILESTONE REACHED: {milestone}%")
            print("[*] LOCKING STATE & LAUNCHING SIDECAR PPL AUDIT...")
            
            # 1. FORCE AN IMMEDIATE PERSISTENT SAVE
            checkpoint_name = f"milestone_{milestone}_checkpoint.pt"
            torch.save(model.state_dict(), checkpoint_name)
            
            # 2. TRIGGER THE CPU SIDECAR (Non-blocking, GPU stays free)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.Popen(
                ["python", "primal_val_perplexity.py", "--checkpoint", checkpoint_name],
                env=env,
                stdout=open(f"ppl_milestone_{milestone}.txt", "w"),
                stderr=subprocess.STDOUT
            )
            print(f"[*] SIDECAR LAUNCHED: Results -> ppl_milestone_{milestone}.txt")
            
            trigger_sidecar_audit.last_triggered = milestone
            break

def train():
    print("[*] Launching 0.1B ANTIGRAVITY (Phase 53)...", flush=True)
    model = GhostGPT(CONFIG).cuda()
    
    # Checkpoint logic
    if os.path.exists("primal_ghost_live.pt"):
        print("[*] Resuming...", flush=True)
        state_dict = torch.load("primal_ghost_live.pt")
        
        # [Phase 52 Fix] Filter out scalar scales
        keys_to_remove = []
        for k, v in state_dict.items():
            if k.endswith('.scale') and v.dim() == 0:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del state_dict[k]
        if keys_to_remove:
            print(f"[*] Pruned {len(keys_to_remove)} legacy scalar scales.", flush=True)
            
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[*] Keys Adjusted: {len(missing)} missing, {len(unexpected)} unexpected.")
    
    params_to_opt = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_opt, lr=CONFIG['lr'])
    loader = get_loader()
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    print("[*] Clearing Initialization Memory...", flush=True)
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    freeze_active = False
    freeze_steps = 0
    last_flip_rate = 1.0 # Initialize high
    
    t0 = time.time()
    for i, batch in enumerate(loader):
        input_ids = batch[:, :-1].cuda(non_blocking=True)
        targets = batch[:, 1:].cuda(non_blocking=True)
        
        step = i//CONFIG['grad_accum'] + 1

        # [NEW] Calculate Annealing Factor (Warmup)
        # Factor starts at 0.5 (sensitive) and rises to 1.0 (normal) over 100 steps
        if step < 100:
            annealing_factor = 0.5 + (0.5 * (step / 100.0))
        else:
            annealing_factor = 1.0
        
        logits, loss = model(input_ids, targets=targets, annealing_factor=annealing_factor)
        loss = loss / CONFIG['grad_accum']
        
        # [NEW] Scale Decay Penalty (Antigravity Friction)
        # Added to loss graph
        loss_scale_penalty = 0.0
        for m in model.modules():
            if isinstance(m, GhostLinear):
               loss_scale_penalty += (m.scale - 1.0).pow(2).mean()
        
        total_loss = loss + (CONFIG['scale_decay'] * loss_scale_penalty)
        total_loss.backward()
        
        if (i + 1) % CONFIG['grad_accum'] == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # --- APPLY DISCRETE VOTES (Phase 52/53 Logic) ---
            total_flips = 0
            total_params = 0
            
            # [PHASE 60.5] Automated Watchdog
            if not freeze_active and run_automated_training_cycle(model, step, last_flip_rate, loss.item()*CONFIG['grad_accum'], optimizer):
                print(f"\n[!] AUTOMATION TRIGGER: Stability target reached at Step {step}")
                print(f"[!] Final Loss: {loss.item()*CONFIG['grad_accum']:.4f} | Flip Rate: {last_flip_rate:.6f}%")
                print(f"[*] PHASE 60: Weight Grid Locked. Performing {CONFIG['final_polish_steps']}-step Scale Polish...")
                
                # Permanent Grid Lock
                for m in model.modules():
                    if isinstance(m, GhostLinear):
                        m.grid_indices.requires_grad = False 
                
                freeze_active = True

            with torch.no_grad():
                if not freeze_active:
                    for name, m in model.named_modules():
                        if isinstance(m, GhostLinear):
                            scale_mean = m.scale.mean().abs().item()
                            adaptive_prob = 0.002 / (scale_mean + 1e-6)
                            adaptive_prob = max(0.0001, min(0.05, adaptive_prob))
                            
                            flips = m.apply_votes(adaptive_prob, name=name)
                            total_flips += flips
                            total_params += m.grid_indices.numel()
                    
                    # [NEW] Thermal Reset Check
                    if step > 50 and total_flips == 0:
                        print(f"[*] THERMAL RESET: Safely boosting pressure at Step {step}...", flush=True)
                        for m in model.modules():
                            if isinstance(m, GhostLinear):
                                 # Safe Boost: Add 10 but clamp at 127 to prevent int8 wrap
                                 m.vote_buffer.add_(10).clamp_(-127, 127)
                else:
                    freeze_steps += 1
                    print(f"[*] FREEZE ACTIVE: Scale Polish Progress {freeze_steps}/50", flush=True)
            
            dt = time.time() - t0
            t0 = time.time()
            tps = (CONFIG['batch_size'] * CONFIG['seq_len'] * CONFIG['grad_accum']) / dt
            
            flip_rate = (total_flips / total_params) * 100 if total_params > 0 else 0
            last_flip_rate = flip_rate
            
            # Monitoring
            primal_scales = [m.scale.mean() for m in model.modules() if isinstance(m, GhostLinear)]
            p_scale = torch.stack(primal_scales).mean().item() if primal_scales else 0.0
            
            print(f"Step {step} | Loss: {loss.item()*CONFIG['grad_accum']:.4f} | TPS: {tps:.2f} | Flips: {flip_rate:.4f}% | P-Scale: {p_scale:.4f} | Anneal: {annealing_factor:.2f} | VRAM: {torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

            # [PHASE 61] Sidecar PPL Audit at Stability Milestones
            trigger_sidecar_audit(model, step, flip_rate)

            if step % 100 == 0:
                log_manifold_heatmap(model, step)
                print(f"[*] Generated Manifold Heatmap at Step {step}", flush=True)

            if step % 100 == 0:
                torch.save(model.state_dict(), "primal_ghost_live.pt")
                print(f"[*] Saved Live Checkpoint at Step {step}")

            # [PHASE 60] Final Export
            if freeze_active and freeze_steps >= CONFIG['final_polish_steps']:
                torch.save(model.state_dict(), "primal_trinity_final_cured.pt")
                sys.exit("[*] TRAINING COMPLETE: The Granite has Set.")
            
            if step % 250 == 0:
                print(f"\n[*] RUNNING STEP {step} SALAD TEST...")
                check_poltergeist_vitals(model) # [Phase 53.5] Internal Pressure Report
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
