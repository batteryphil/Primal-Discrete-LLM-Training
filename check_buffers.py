import torch
import os

checkpoint_path = "v5_6_coder_live.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    interesting_buffers = [
        'trm_core.voltage_boost',
        'trm_core.tamer_threshold',
        'trm_core.current_step',
        'trm_core.last_entropy'
    ]
    
    for name, param in state_dict.items():
        if any(b in name for b in interesting_buffers):
            print(f"{name}: {param.item()}")
else:
    print(f"Checkpoint {checkpoint_path} not found.")
