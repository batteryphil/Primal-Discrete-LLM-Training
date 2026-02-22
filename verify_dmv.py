import torch
from primal_train_modular import GhostGPT, AntigravitySentinelV56, CONFIG, PRIMAL_LUT
from ghost_core import GhostLinearTandem

def test_dmv_step():
    print("[*] Initializing model with DMV...")
    model = GhostGPT(CONFIG).cuda()
    
    base_params = [p for n, p in model.named_parameters() if '_idx' not in n]
    optimizer = torch.optim.AdamW(base_params, lr=1e-4)
    
    sentinel = AntigravitySentinelV56(model, optimizer)
    
    print("[*] Running forward/backward pass...")
    # Mock input
    x = torch.randint(0, CONFIG['vocab_size'], (1, CONFIG['seq_len'])).cuda()
    targets = torch.randint(0, CONFIG['vocab_size'], (1, CONFIG['seq_len'])).cuda()
    
    logits, loss = model(x, targets=targets)
    loss.backward()
    
    print("[*] Applying DMV step...")
    success, norm, flip_rate = sentinel.apply_safestep_and_vote()
    
    print(f"[*] Success: {success}")
    print(f"[*] Total Norm: {norm.item():.4f}")
    print(f"[*] Flip Rate: {flip_rate:.4f}%")
    
    # Check if velocity buffer is populated
    if sentinel.dmv.velocity_buffer:
        print("[*] Velocity buffer successfully initialized and populated.")
    else:
        print("[!] Velocity buffer is empty!")

if __name__ == "__main__":
    test_dmv_step()
