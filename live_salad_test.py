import torch
from primal_train_ghost import GhostGPT, CONFIG, check_poltergeist_vitals as calculate_pressure
import os

def live_salad_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing model on {device}...")
    model = GhostGPT(CONFIG).to(device)
    
    checkpoint_path = "primal_ghost_live.pt"
    if os.path.exists(checkpoint_path):
        print(f"[*] Loading live checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("[!] Live checkpoint not found.")
        return

    print("\n" + "="*50)
    print("POLTERGEIST V3: LIVE INTERNAL PRESSURE REPORT")
    print("="*50)
    calculate_pressure(model)

if __name__ == "__main__":
    live_salad_test()
