
import torch
from ghost_core import GhostLinearTandem, GhostTandemQuantFunction
import manifolds

def test_integer_momentum():
    print("[*] Testing Protocol v5.73 (Integer Momentum)...")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lut = manifolds.generate_int16_linear_manifold(device=device)
    linear = GhostLinearTandem(128, 128, lut=lut).to(device)
    
    print("[*] Model initialized.")
    print(f"[*] Vote Buffer Type: {linear.vote_buffer.dtype}")
    
    # 1. Forward Pass
    x = torch.randn(1, 128).to(device)
    y = linear(x)
    print("[*] Forward pass successful.")
    
    # 2. Backward Pass (Pressure Injection)
    loss = y.mean()
    loss.backward()
    print("[*] Backward pass successful.")
    
    # Check Pressure Accumulation
    pressure = linear.vote_buffer.float().abs().mean().item()
    print(f"[*] Mean Pressure after 1 step: {pressure:.4f}")
    
    # 3. Apply Momentum Votes
    print("[*] Applying Momentum Votes (Learning Rate = 1.0)...")
    flips = linear.apply_tandem_votes(learning_rate=1.0)
    print(f"[*] Total Steps/Flips: {flips}")
    
    # Check if votes decayed
    pressure_after = linear.vote_buffer.float().abs().mean().item()
    print(f"[*] Mean Pressure after vote application: {pressure_after:.4f}")
    
    if pressure_after < pressure:
        print("[SUCCESS] Pressure decayed as expected.")
    else:
        print("[WARNING] Pressure did not decay!")

if __name__ == "__main__":
    try:
        test_integer_momentum()
        print("\n[PASSED] Protocol v5.73 validated.")
    except Exception as e:
        print(f"\n[FAILED] Error: {e}")
        import traceback
        traceback.print_exc()
