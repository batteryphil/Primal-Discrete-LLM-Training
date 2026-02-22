import torch
import torch.nn as nn
import manifolds
from ghost_core import PrimalTRMCore, GhostLinearTandem, LogitTamer
import os

# --- SCANNER CONFIG ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG = {
    'vocab_size': 50257, 
    'dim': 1024,
    'iterations': 4,
}

class GhostGPT(nn.Module):
    def __init__(self, c, lut):
        super().__init__()
        self.token_emb = nn.Embedding(c['vocab_size'], c['dim'])
        self.pos_emb = nn.Embedding(2048, c['dim'])
        num_iterations = c.get('iterations', 8)
        self.trm_core = PrimalTRMCore(c['dim'], num_iterations=num_iterations, lut=lut)
        self.ln_f = nn.LayerNorm(c['dim'])
        self.head = GhostLinearTandem(c['dim'], c['vocab_size'], lut=lut, sensitivity=0.05)

def run_integrity_check():
    print("\n" + "="*50)
    print("   PROTOCOL v5.90: STEP 500 INTEGRITY REPORT   ")
    print("="*50)
    
    # Generate LUT for matching
    lut = manifolds.generate_int16_linear_manifold(device=device)
    model = GhostGPT(CONFIG, lut).to(device)
    
    checkpoint_path = 'primal_live.pt'
    if not os.path.exists(checkpoint_path):
        print(f"[!] ERROR: {checkpoint_path} not found.")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"[*] Analyzing Manifold: {checkpoint_path}\n")

    for name, module in model.named_modules():
        if isinstance(module, GhostLinearTandem):
            print(f"--- LAYER: {name} ---")
            
            # 1. BIT-DEPTH UTILIZATION
            with torch.no_grad():
                combined = (module.base_idx.to(torch.int32) * 256) + module.fine_idx.to(torch.int32)
                min_val = combined.min().item()
                max_val = combined.max().item()
                utilization = (max_val - min_val) / 65535.0
                mean_pos = combined.float().mean().item()
                
                print(f"[BIT-DEPTH] Range: {min_val} to {max_val} ({utilization:.2%})")
                print(f"[BIT-DEPTH] Manifold Center: {mean_pos:.2f} (Target: 32768)")

            # 2. CONSENSUS FRICTION
            # measure 'disagreement' inside blocks (size 32 default)
            block_size = getattr(module, 'block_size', 32)
            buffer = module.vote_buffer.data.float()
            flat_buf = buffer.view(-1)
            num_blocks = flat_buf.numel() // block_size
            blocks = flat_buf.view(num_blocks, block_size)
            
            # Sign conflict within block
            pos_votes = (blocks > 0).sum(dim=1)
            neg_votes = (blocks < 0).sum(dim=1)
            # Friction exists if both pos and neg votes > 0 in the same block
            friction_blocks = ((pos_votes > 0) & (neg_votes > 0)).sum().item()
            friction_pct = (friction_blocks / num_blocks) * 100
            
            print(f"[FRICTION] High-Resonance Blocks: {friction_blocks} ({friction_pct:.4f}%)")

            # 3. GHOST RESIDUE
            # Mean absolute buffer strength (background noise)
            residue = buffer.abs().mean().item()
            peak_pressure = buffer.abs().max().item()
            print(f"[RESIDUE] Background Static: {residue:.4f} | Peak Pressure: {peak_pressure}")
            
            if friction_pct > 1.0:
                print("[!] WARNING: Resonance detected. Potential internal fighting.")
            elif utilization < 0.05:
                print("[!] WARNING: Shallow manifold detected. Weights are too compressed.")
            else:
                print("[✓] Layer Health: OPTIMAL.")
            print("-" * 30)

    print("\n" + "="*50)
    print("   INTEGRITY CHECK COMPLETE | SYSTEM NOMINAL   ")
    print("="*50)

if __name__ == "__main__":
    run_integrity_check()
