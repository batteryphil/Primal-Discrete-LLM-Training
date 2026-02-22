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
    print("[*] Generating 8-bit Prime-Power Manifold (Antigravity v3.2)...")
    
    # 1. THE PRIME-POWER NUCLEUS
    # We use 1 / (p^1.15) to shift density slightly outward from zero
    # while maintaining a clustered distribution near sigma=0.02.
    core_values = set()
    core_values.add(0.0)
    
    # Primes up to ~500 provide a good spread
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    # Add more primes to fill density
    extended_primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179]
    all_primes = primes + extended_primes
    
    for p in all_primes:
        # Prime-Power shift: 1 / p^1.15
        val = 1.0 / (math.pow(p, 1.15))
        core_values.add(val)
        core_values.add(-val)
        
    # 2. THE SIGNAL CLUSTERS (Harmonic Harmonics)
    # Target resolution around sigma=0.02 and 0.1
    for i in range(2, 40):
        val = 1.0 / (i * 2.5) # Spreading out
        core_values.add(val)
        core_values.add(-val)

    # 3. THE TAILS (Energy Capture)
    tail_values = set()
    tail_values.add(1.0)
    tail_values.add(-1.0)
    
    # Linear stepping for high-magnitude capture
    for i in range(1, 32):
        val = i / 32.0 
        tail_values.add(val)
        tail_values.add(-val)
        
    full_set = core_values.union(tail_values)
    lut_list = sorted(list(full_set))
    
    # Adjust to exactly 256
    current_len = len(lut_list)
    if current_len < 256:
        while len(lut_list) < 256:
            diffs = [lut_list[i+1] - lut_list[i] for i in range(len(lut_list)-1)]
            idx = diffs.index(max(diffs))
            midpoint = (lut_list[idx] + lut_list[idx+1]) / 2
            lut_list.insert(idx+1, midpoint)
    elif current_len > 256:
        while len(lut_list) > 256:
            diffs = [lut_list[i+1] - lut_list[i] for i in range(len(lut_list)-1)]
            min_diff = min(diffs)
            idx = diffs.index(min_diff)
            # Favor removing points further from 0.02 to keep core "vision"
            lut_list.pop(idx+1)

    print(f"[*] Final v3.2 Manifold Size: {len(lut_list)}")
    return torch.tensor(lut_list, device=device, dtype=torch.float32)

def generate_linear_manifold(device='cpu'):
    """Generates a fully populated, evenly spaced 8-bit linear grid (-1.0 to 1.0)."""
    print("[*] Generating 8-bit Linear Manifold (Ghost-TTS Hotfix)...")
    # 256 evenly spaced values between -1.0 and 1.0
    lut = torch.linspace(-1.0, 1.0, 256, device=device)
    return lut

def generate_fp8_manifold(device='cpu'):
    """Generates the 256 values of the FP8 E4M3FN standard."""
    print("[*] Generating 8-bit FP8 (E4M3FN) Manifold...")
    values = []
    # E4M3FN: 1 Sign, 4 Exponent, 3 Mantissa. Bias = 7.
    # E=0: Denormal. E=1..15: Normal. M=111 at E=15 is NaN (skip).
    
    for s in [0, 1]:
        sign = -1.0 if s == 1 else 1.0
        
        for e in range(16): # 4 bits
            for m in range(8): # 3 bits
                
                # Check for NaN (S.1111.111)
                if e == 15 and m == 7:
                    continue # Skip NaN
                
                if e == 0:
                    # Denormal: (-1)^S * 2^(-6) * (M / 8)
                    val = sign * (2.0 ** -6) * (m / 8.0)
                else:
                    # Normal: (-1)^S * 2^(E - 7) * (1 + M / 8)
                    val = sign * (2.0 ** (e - 7)) * (1.0 + (m / 8.0))
                
                values.append(val)
                
    # Sort and convert to tensor
    values = sorted(list(set(values))) # Remove duplicates (like +0, -0)
    
    # Pad to 256 if needed (usually 254 unique non-NaN values + maybe duplicates for +/-0)
    # E4M3FN has 254 unique values (NaNs removed). We can duplicate 0 or max to fill to 256.
    while len(values) < 256:
        values.append(values[-1]) # Pad with max value
        
    print(f"[*] FP8 Manifold Generated. Size: {len(values)}")
    return torch.tensor(values, device=device, dtype=torch.float32)

def generate_int16_linear_manifold(device='cpu'):
    """Generates a high-fidelity 16-bit linear manifold (65,536 steps)."""
    print("[*] Generating 16-bit Linear Manifold (Protocol v5.0)...")
    # 65,536 evenly spaced values between -1.0 and 1.0
    lut = torch.linspace(-1.0, 1.0, 65536, device=device)
    return lut

def generate_int16_prime_manifold(device='cpu'):
    """Generates a high-fidelity 16-bit Prime-Harmonic manifold (65,536 steps)."""
    print("[*] Generating 16-bit Prime-Harmonic Manifold (Protocol v6.00)...")
    
    values = set()
    values.add(0.0)
    
    # 1. THE PRIME-POWER NUCLEUS
    # Use primes up to ~50,000 for high resolution
    limit = 60000
    is_prime = [True] * (limit + 1)
    primes = []
    for p in range(2, limit + 1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
    
    for p in primes:
        val = 1.0 / (math.pow(p, 1.05))
        if val <= 1.0:
            values.add(val)
            values.add(-val)
            
    # 2. THE SIGNAL CLUSTERS
    for i in range(2, 10000):
        val = 1.0 / (i * 1.2)
        if val <= 1.0:
            values.add(val); values.add(-val)

    # 3. THE TAILS
    for i in range(1, 8192):
        val = i / 8192.0
        values.add(val); values.add(-val)
        
    values.add(1.0); values.add(-1.0)
    lut_list = sorted(list(values))
    
    # Adjust to exactly 65536
    current_len = len(lut_list)
    if current_len < 65536:
        print(f"[*] Interpolating {65536 - current_len} points...")
        while len(lut_list) < 65536:
            diffs = [lut_list[i+1] - lut_list[i] for i in range(len(lut_list)-1)]
            idx = diffs.index(max(diffs))
            midpoint = (lut_list[idx] + lut_list[idx+1]) / 2
            lut_list.insert(idx+1, midpoint)
    elif current_len > 65536:
        print(f"[*] Pruning {current_len - 65536} points...")
        while len(lut_list) > 65536:
            diffs = [lut_list[i+1] - lut_list[i] for i in range(len(lut_list)-1)]
            min_diff = min(diffs)
            idx = diffs.index(min_diff)
            lut_list.pop(idx+1)

    print(f"[*] 16-bit Prime Manifold Generated. Size: {len(lut_list)}")
    return torch.tensor(lut_list, device=device, dtype=torch.float32)

if __name__ == "__main__":
    lut = generate_manifold()
    print("Sample Values (Center):", lut[120:136])
    print("Sample Values (Ends):", lut[:5], lut[-5:])
    
    lut16 = generate_int16_prime_manifold()
    print("16-bit Manifold Size:", len(lut16))
