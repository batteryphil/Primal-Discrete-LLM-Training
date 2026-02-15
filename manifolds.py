import torch
import math

# ==============================================================================
# PRIME-HARMONIC MANIFOLD GENERATOR (8-Bit)
# ==============================================================================
# Generates a 256-value Look-Up Table (LUT) for Discrete Optimization.
# Architecture:
# - 192 "Dense" slots: harmonic reciprocals (1/2, 1/3, 1/5...) tailored for fine gradients.
# - 64 "Outlier" slots: linear/exp tails for large gradient captures.
# ==============================================================================

def generate_manifold(device='cpu'):
    print("[*] Generating 8-bit Prime-Harmonic Manifold...")
    
    # 1. THE CORE (Dense Reciprocals) - 192 Values
    # We want a dense cluster around 0.0 to capture subtle changes.
    # Pattern: +/- 1/n where n goes from 2 to 97 (approx)
    
    core_values = set()
    core_values.add(0.0) # The Void
    
    # Harmonic Series: 1/2, 1/3, 1/4... up to 1/97
    for i in range(2, 98):
        val = 1.0 / i
        core_values.add(val)
        core_values.add(-val)
        
    # Prime Harmonics (Reinforcement)
    # 1/p^2 for p in primes, to add zest near zero
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in primes:
        val = 1.0 / (p * 1.5) # slightly shifted
        core_values.add(val)
        core_values.add(-val)

    # 2. THE TAILS (Outliers) - 64 Values
    # We need to capture 1.0, 2.0 (maybe?), and fractions like 0.666 (The Bridge)
    tail_values = set()
    tail_values.add(1.0)
    tail_values.add(-1.0)
    tail_values.add(0.666666) # 2/3 Bridge
    tail_values.add(-0.666666)
    
    # Linear stepping from 0.1 to 1.0 filling gaps
    for i in range(1, 32):
        val = i / 32.0 
        tail_values.add(val)
        tail_values.add(-val)
        
    # Combine
    full_set = core_values.union(tail_values)
    
    # Sort
    lut_list = sorted(list(full_set))
    
    # Check size
    current_len = len(lut_list)
    print(f"[*] Initial Manifold Size: {current_len}")
    
    # Adjust to exactly 256
    if current_len < 256:
        # Interpolate to fill
        needed = 256 - current_len
        print(f"[*] Interpolating {needed} values...")
        # Add midpoints of largest gaps
        while len(lut_list) < 256:
            diffs = [lut_list[i+1] - lut_list[i] for i in range(len(lut_list)-1)]
            max_diff = max(diffs)
            idx = diffs.index(max_diff)
            midpoint = (lut_list[idx] + lut_list[idx+1]) / 2
            lut_list.insert(idx+1, midpoint)
            
    elif current_len > 256:
        # Prune closest neighbors
        to_remove = current_len - 256
        print(f"[*] Pruning {to_remove} values...")
        # Naive pruning: remove extremely close values
        # In a real scenario, we'd be smarter, but let's just trim ends or middle density
        # Loop verify
        while len(lut_list) > 256:
            diffs = [lut_list[i+1] - lut_list[i] for i in range(len(lut_list)-1)]
            min_diff = min(diffs)
            idx = diffs.index(min_diff)
            # Remove one of the pair
            lut_list.pop(idx+1)

    print(f"[*] Final Manifold Size: {len(lut_list)}")
    
    # Convert to Tensor
    tensor_lut = torch.tensor(lut_list, device=device, dtype=torch.float32)
    return tensor_lut

def generate_linear_manifold(device='cpu'):
    """Generates a fully populated, evenly spaced 8-bit linear grid (-1.0 to 1.0)."""
    print("[*] Generating 8-bit Linear Manifold (Ghost-TTS Hotfix)...")
    # 256 evenly spaced values between -1.0 and 1.0
    lut = torch.linspace(-1.0, 1.0, 256, device=device)
    return lut

if __name__ == "__main__":
    lut = generate_manifold()
    print("Sample Values (Center):", lut[120:136])
    print("Sample Values (Ends):", lut[:5], lut[-5:])
