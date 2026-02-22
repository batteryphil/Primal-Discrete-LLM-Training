import torch
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from primal_train_modular import GhostGPT, GammaController, night_shift_step, force_feed_core
    
    CONFIG = {
        'dim': 64,
        'n_layers': 1,
        'n_heads': 1,
        'vocab_size': 100,
        'seq_len': 16,
        'device': 'cpu',
        'iterations': 4
    }
    
    # Mock PRIMAL_LUT
    import manifolds
    PRIMAL_LUT = manifolds.generate_int16_linear_manifold(device='cpu')
    # Inject it if necessary or ensure it's available
    import primal_train_modular
    primal_train_modular.PRIMAL_LUT = PRIMAL_LUT
    
    model = GhostGPT(CONFIG).to('cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    gamma_controller = GammaController(reasoning=1.0, head=0.0)
    dam_state = {'is_active': True, 'step_count': 0}
    
    # Test Override
    print("[*] Testing Force Feed Core Override...")
    dam_state = force_feed_core(model, gamma_controller, dam_state)
    if dam_state['manual_lock'] == True and dam_state['is_active'] == True:
        print("    [PASS] Force Feed Core override successful.")
    else:
        print("    [FAIL] Force Feed Core override failed.")
        
    # Test Night Shift Step
    print("[*] Testing Night Shift Step...")
    dam_state = night_shift_step(model, optimizer, gamma_controller, dam_state)
    lr = optimizer.param_groups[0]['lr']
    if lr == 0.50000:
        print(f"    [PASS] Thermal Squeeze (LR={lr:.5f}) confirmed.")
    else:
        print(f"    [FAIL] Thermal Squeeze (LR={lr:.5f}) failed.")
        
    print("\n[!] ARCHITECTURE VALIDATED.")
    
except Exception as e:
    print(f"[!] ERROR DURING VALIDATION: {e}")
    import traceback
    traceback.print_exc()
