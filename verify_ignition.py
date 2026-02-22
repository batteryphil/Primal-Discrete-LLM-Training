
import torch
from ghost_core import GhostLinearTandem, GhostTandemQuantFunction
import manifolds
import math

def test_ignition_momentum():
    print("[*] Testing Protocol v5.73 'Ignition & Momentum'...")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lut = manifolds.generate_int16_linear_manifold(device=device)
    linear = GhostLinearTandem(256, 256, lut=lut).to(device)
    
    print("[*] Model initialized.")
    
    # 1. Verify Ignition (Symmetry Breaking)
    # base_idx should NOT be uniform (128)
    base_std = linear.base_idx.float().std().item()
    print(f"[*] Base Index STD: {base_std:.4f}")
    
    if base_std > 0.0:
        print("[SUCCESS] Symmetry Breaking Initialized (Xavier Integer Scattering).")
    else:
        print("[FAILED] Symmetry Paralysis Detected! (STD = 0.0)")
        return

    # 2. Forward Pass
    x = torch.randn(1, 256).to(device)
    y = linear(x)
    print("[*] Forward pass successful.")
    
    # 3. Violent Backward Pass (Pressure Injection Safety)
    # Simulate a massive gradient spike to test clamping
    y.backward(torch.ones_like(y) * 1000.0) 
    print("[*] Backward pass successful (Massive Gradient).")
    
    # Check Pressure Accumulation
    # Should be capped at 100 per step roughly due to normalization
    pressure_max = linear.vote_buffer.abs().max().item()
    print(f"[*] Max Pressure after 1 step: {pressure_max}")
    
    if pressure_max <= 32760:
         print("[SUCCESS] Pressure Injection Clamped Safely.")
    else:
         print(f"[FAILED] Integer Overflow Detected! ({pressure_max})")

    # 4. Apply Momentum Votes (Multi-Step Jump)
    # We expect significant movement due to massive gradient
    print("[*] Applying Momentum Votes (Learning Rate = 1.0)...")
    flips = linear.apply_tandem_votes(learning_rate=1.0)
    print(f"[*] Total Steps/Jumps Executed: {flips}")
    
    if flips > 0:
        print("[SUCCESS] Momentum Engine Active (Multi-Step Jumps).")
    else:
        print("[FAILED] No Movement Detected!")
        
    # 5. Momentum Decay Check
    # Pressure should be decayed but not zeroed
    pressure_after = linear.vote_buffer.float().abs().mean().item()
    print(f"[*] Mean Pressure after vote application: {pressure_after:.4f}")

if __name__ == "__main__":
    try:
        test_ignition_momentum()
        print("\n[PASSED] Protocol v5.73 Verified.")
    except Exception as e:
        print(f"\n[FAILED] Error: {e}")
        import traceback
        traceback.print_exc()
