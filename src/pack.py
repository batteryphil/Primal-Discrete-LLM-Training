import torch
import torch.nn as nn
import os
import struct

checkpoint_path = "trinity_evolved_500steps.pt"
output_path = "trinity_1.58bit_packed.bin"
device = "cpu"

print(f"--- Project Trinity: Final Packing Protocol ---")

def pack_ternary_weights(weight_tensor):
    # Quantize to signs: -1, 0, 1
    w_sign = torch.sign(weight_tensor.flatten())
    # Map to unsigned: 0, 1, 2
    w_int = (w_sign + 1).to(torch.int8)
    
    # Ensure divisible by 4 for byte alignment (2 bits * 4 = 8 bits)
    padding = (4 - (len(w_int) % 4)) % 4
    if padding > 0:
        w_int = torch.cat([w_int, torch.zeros(padding, dtype=torch.int8)])
    
    w_groups = w_int.view(-1, 4)
    # Pack 4 weights per byte: [w0][w1][w2][w3]
    packed_byte = (w_groups[:, 0] << 6) | (w_groups[:, 1] << 4) | (w_groups[:, 2] << 2) | w_groups[:, 3]
    return packed_byte.to(torch.uint8)

if not os.path.exists(checkpoint_path):
    print(f"ERROR: Checkpoint {checkpoint_path} not found.")
    exit()

print(f"Loading weights from {checkpoint_path}...")
state_dict = torch.load(checkpoint_path, map_location=device)

original_size_bytes = os.path.getsize(checkpoint_path)
print(f"Original Checkpoint Size: {original_size_bytes / (1024**2):.2f} MB")

packed_total = 0
with open(output_path, "wb") as f:
    f.write(b'TRIN') # Header
    for name, param in state_dict.items():
        # Only pack linear weights (not biases, norms, or embeddings for this demo)
        if "weight" in name and "LayerNorm" not in name and "embed" not in name and len(param.shape) == 2:
            packed_data = pack_ternary_weights(param)
            # Write length header for this block
            f.write(struct.pack('I', len(packed_data)))
            f.write(packed_data.numpy().tobytes())
            packed_total += len(packed_data)
            print(f"Packed {name}: {param.shape}")

final_size_bytes = os.path.getsize(output_path)
print(f"\n--- PACKING COMPLETE ---")
print(f"Final Packed Size: {final_size_bytes / (1024**2):.2f} MB")
print(f"Compression Ratio (vs FP16 Checkpoint): {original_size_bytes / final_size_bytes:.2f}x")
print(f"Effective Bits Per Parameter (Packed Region): 2.0 bits")
