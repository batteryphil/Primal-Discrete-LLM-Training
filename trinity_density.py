"""Trinity Density Diagnostic — Fast GPU check."""
import torch, time
print("[*] Loading model for density check...", flush=True)
t0 = time.time()
from primal_train_ghost import GhostGPT, CONFIG
print(f"[*] Import took {time.time()-t0:.1f}s", flush=True)

t1 = time.time()
model = GhostGPT(CONFIG).cuda()
sd = torch.load("primal_ghost_live.pt")
model.load_state_dict(sd, strict=False)
print(f"[*] Model loaded in {time.time()-t1:.1f}s", flush=True)

active_levels = set()
for m in model.modules():
    if hasattr(m, 'grid_indices'):
        active_levels.update(m.grid_indices.unique().tolist())
print(f"\n--- TRINITY DENSITY: {len(active_levels)}/256 levels active ---")
print(f"--- Effective bits per weight: {torch.log2(torch.tensor(float(len(active_levels)))):.2f} ---")
