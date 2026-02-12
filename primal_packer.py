import struct
import numpy as np
import os

def pack_weights_v2(weights_fp32, scale):
    """
    Packs FP32 weights into 4-bit nibbles with 128-bit (16-byte) row alignment.
    """
    rows, cols = weights_fp32.shape
    
    # 1. Pad columns to multiple of 32 (32 weights = 16 bytes = 1 uint4)
    # This ensures every row starts on a 16-byte boundary (if base is aligned)
    # and every row is a multiple of 16 bytes long.
    remainder = cols % 32
    if remainder != 0:
        padding = 32 - remainder
        # Pad with zeros (neutral weight)
        weights_fp32 = np.pad(weights_fp32, ((0,0), (0, padding)), mode='constant', constant_values=0)
        cols += padding
        print(f"  [Packer] Padded cols to {cols} (multiple of 32)")

    # 2. Quantize (Primitive / Dummy logic for packer - real logic is in export script)
    # This function assumes weights are already scaled or we just map them?
    # For this packer utility, we assume we just need to pack INDICES.
    # But usually we pack mapped values.
    # Let's assume input 'weights_fp32' are actually the Quantized Integers (0-15) for this snippet,
    # or we do the simple mapping here. 
    # Let's map strict 0-15 just to show the packing logic.
    
    # We'll just assume input is standard float and we map closest for demonstration.
    # In real pipeline, 'primal_export_v4.py' handles the quantization math.
    # This function focuses on the PACKING BUFFER layout.
    
    # ... quantization logic omitted (assume we have uint8 indices 0..15) ...
    # For demo, creating dummy indices
    indices = np.zeros((rows, cols), dtype=np.uint8) 
    # Fill with pattern
    indices[:] = 0xA # Pattern 1010
    
    # 3. Pack Nibbles
    # 2 weights per byte.
    # byte[i] = (w[2*i] << 4) | w[2*i+1]
    
    # Reshape to (rows, cols/2, 2)
    # But wait, packed format: High Nibble is first?
    # W[0] -> High, W[1] -> Low.
    # byte = (W0 << 4) | W1
    
    indices_reshaped = indices.reshape(rows, cols // 2, 2)
    packed_bytes = (indices_reshaped[:, :, 0] << 4) | indices_reshaped[:, :, 1]
    
    return packed_bytes.astype(np.uint8).tobytes(), rows, cols

def export_model(filename="model_aligned.primal"):
    print(f"Exporting {filename} with 128-bit Alignment...")
    with open(filename, 'wb') as f:
        f.write(b'PRML')
        
        # Example Layer
        rows, cols = 768, 3072 # Aligned
        # Create dummy weights
        w = np.random.randn(rows, cols).astype(np.float32)
        packed_data, r, c = pack_weights_v2(w, 1.0)
        
        # Write Name
        name = b"layer_0.weight"
        f.write(struct.pack('I', len(name)))
        f.write(name)
        
        # Write Header
        f.write(struct.pack('I', r))
        f.write(struct.pack('I', c))
        f.write(struct.pack('f', 1.0)) # scale
        f.write(struct.pack('I', len(packed_data)))
        
        f.write(packed_data)
        
    print("Done.")

if __name__ == "__main__":
    export_model()
