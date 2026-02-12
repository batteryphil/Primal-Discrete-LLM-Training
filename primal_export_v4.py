import torch
import struct
import os
from primal_train_v2 import PrimalBrain, LUT

def export_trained_primal(checkpoint_path, output_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    print(f"Loading trained weights from {checkpoint_path}...")
    # Model init arguments must match primal_train_v2 defaults or be explicit
    # V2 Defaults: n_layer=12, n_embd=384, n_head=8
    model = PrimalBrain(n_layer=12, n_embd=384, n_head=8)
    try:
        model.load_state_dict(torch.load(checkpoint_path))
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return
        
    model.eval()

    with open(output_path, "wb") as f:
        f.write(b"PRML") # Magic Header
        
        for name, param in model.named_parameters():
            # Skip embeddings (handled by Python/HF)
            if "embedding" in name:
                print(f"Skipping Embedding Layer: {name}")
                continue
                
            if "weight" in name and param.dim() == 2:
                print(f"Exporting: {name} {param.shape}")
                # Use the scale the model learned during training
                scale = param.abs().max().item() + 1e-6
                
                # Snap to grid
                w_norm = param.data / scale
                
                # Expand for broadcasting
                # w_norm: [Rows, Cols, 1]
                # LUT: [16]
                diff = torch.abs(w_norm.unsqueeze(-1) - LUT.to(w_norm.device))
                indices = torch.argmin(diff, dim=-1).cpu().numpy().astype('uint8')
                
                # Pack 4-bit
                if indices.size % 2 != 0:
                    print(f"Warning: Tensor {name} has odd size {indices.size}, padding not implemented.")
                    
                packed = (indices.flat[0::2] << 4) | (indices.flat[1::2] & 0x0F)
                packed_data = packed.tobytes()
                
                # Write Header & Data
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('II f I', param.shape[0], param.shape[1], scale, len(packed_data)))
                f.write(packed_data)
                
    print(f"Export Success: {output_path}")

if __name__ == "__main__":
    export_trained_primal("primal_expanse.pt", "primal_expanse.primal")
