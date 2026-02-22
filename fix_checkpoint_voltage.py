import torch
import os

checkpoint_path = "v5_6_coder_live.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # It's a full dict
        checkpoint['model_state_dict']['trm_core.voltage_boost'] = torch.tensor(1.0)
    else:
        # It's a bare state_dict
        checkpoint['trm_core.voltage_boost'] = torch.tensor(1.0)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"[SUCCESS] trm_core.voltage_boost reset to 1.0 in {checkpoint_path}")
else:
    print(f"Checkpoint {checkpoint_path} not found.")
