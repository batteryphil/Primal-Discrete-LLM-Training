import torch
import sys

def inspect_checkpoint(path):
    print(f"\n--- Inspecting {path} ---")
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict):
            print(f"Keys: {list(checkpoint.keys())}")
            if 'step' in checkpoint:
                print(f"Training Step: {checkpoint['step']}")
            elif 'current_step' in checkpoint:
                print(f"Current Step: {checkpoint['current_step']}")
            
            # Check for trm_core.current_step in state_dict if it's there
            sd = checkpoint.get('model_state_dict', checkpoint)
            if 'trm_core.current_step' in sd:
                print(f"Model Buffer current_step: {sd['trm_core.current_step'].item()}")
        else:
            print(f"Checkpoint is of type {type(checkpoint)}")
    except Exception as e:
        print(f"Load failed: {e}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect_checkpoint(p)
