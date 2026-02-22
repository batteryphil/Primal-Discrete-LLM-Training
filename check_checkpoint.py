import torch
import os

checkpoint_path = "v5_6_coder_live.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    for name, param in state_dict.items():
        if torch.isnan(param).any():
            print(f"[!!!] {name} contains NaNs")
        else:
            print(f"[OK] {name} is clean. Max: {param.max().item():.4f}, Min: {param.min().item():.4f}")
else:
    print(f"Checkpoint {checkpoint_path} not found.")
