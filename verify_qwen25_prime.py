import torch
from manifolds import generate_int16_prime_manifold
from qwen25_prime_wrapper import PrimeQwen2CausalLM

def verify():
    config = {
        'hidden_size': 1536,
        'num_hidden_layers': 28,
        'num_attention_heads': 12,
        'num_key_value_heads': 2,
        'intermediate_size': 8960,
        'vocab_size': 151936,
        'rms_norm_eps': 1e-6,
        'rope_theta': 1000000,
        'max_position_embeddings': 2048
    }

    print("[*] Generating 16-bit Prime-Harmonic Manifold...")
    lut = generate_int16_prime_manifold(device='cpu')

    print("[*] Initializing model shell...")
    model = PrimeQwen2CausalLM(config, lut=lut)
    
    try:
        print("[*] Loading checkpoint...")
        model.load_state_dict(torch.load("qwen25_coder_prime_init.pt", map_location='cpu'))
        print("[✓] Checkpoint loaded.")
    except Exception as e:
        print(f"[!] Warning: Could not load checkpoint (Expected if importer hasn't run yet): {e}")
        return

    model.eval()
    
    # Check weight distribution
    print("[*] Checking base_idx distribution (Layer 0)...")
    base_idx = model.layers[0].self_attn.q_proj.base_idx
    mean_val = base_idx.float().mean().item()
    std_val = base_idx.float().std().item()
    print(f"  Q_Proj Mean: {mean_val:.2f} | Std: {std_val:.2f}")

    # Forward Pass Test
    print("[*] Running Forward Pass Test...")
    dummy_ids = torch.randint(0, config['vocab_size'], (1, 16))
    with torch.no_grad():
        logits, _ = model(dummy_ids)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    
    if torch.isnan(logits).any():
        print("[!] ERROR: NaNs detected in logits.")
    else:
        print("[✓] Forward pass successful.")

if __name__ == "__main__":
    verify()
