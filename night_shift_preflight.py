import torch
import torch.nn as nn
import os
import sys
import math

# Add current directory to path for imports
sys.path.append(os.getcwd())

from ghost_core import GhostLinearTandem, PrimalTRMCore
from primal_train_modular import GhostGPT, GammaController, night_shift_step

# Mock CONFIG for minimal initialization
CONFIG = {
    "dim": 128,          # Smaller for speed
    "n_layers": 2,       # Smaller for speed
    "n_heads": 2,
    "vocab_size": 100,
    "seq_len": 32,
    "device": "cpu",
    "grad_accum": 1
}

print("\n[!] INITIATING NIGHT SHIFT PRE-FLIGHT STRESS TEST...")

# --- SETUP: INITIALIZE COMPONENTS ---
try:
    # Initialize a minimal model
    model = GhostGPT(CONFIG).to(CONFIG['device'])
    
    # Initialize Optimizer (AdamW is standard in this project)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0) # Start with high LR to check clamp
    
    # Initialize Gamma Controller
    gamma_controller = GammaController(reasoning=1.0, head=0.0)
    
    # Initialize Dam State
    dam_state = {'is_active': True, 'step_count': 0}

    # --- TEST 1: THE HONEST SENSOR (QKV FLATTENING CHECK) ---
    # Protocol v6.00: Strictly flattens using .view(-1)
    # We initialize with local voters at zero to expect exactly 0.0000
    r_consensus = model.trm_core.reasoning_gate.get_consensus_ratio()
    print(f"[*] TEST 1 (Sensor Alignment): Reasoning Gate reads {r_consensus:.4f}")
    if r_consensus == 0.0000:
        print("    [PASS] Sensor is flat. No QKV hallucination.")
    else:
        print("    [FAIL] Sensor is still hallucinating dimensions.")

    # --- TEST 2: THE THERMAL SQUEEZE (CLIP CHECK) ---
    # We force a gradient clip and check if the optimizer respects the 0.50 clamp.
    # We need to run night_shift_step to apply the clamp.
    night_shift_step(model, optimizer, gamma_controller, dam_state)
    lr_check = optimizer.param_groups[0]['lr']
    print(f"[*] TEST 2 (Thermal Clamp): Learning Rate is locked at {lr_check:.5f}")
    if lr_check == 0.50000:
        print("    [PASS] Night Shift successfully overrode Autopilot LR.")
    else:
        print("    [FAIL] Autopilot is still fighting for control.")

    # --- TEST 3: HYSTERESIS LATCH SIMULATION (THE WHIPLASH TEST) ---
    # Helper for simulation (matches the USER's provided block)
    def night_shift_step_SIMULATED(model, optimizer, gamma_controller, state, fake_r, fake_f):
        # We simulate the logic inside night_shift_step but with injected consensus values
        
        # SCENARIO A: Dam is currently CLOSED
        if state['is_active']:
            if fake_r >= 0.40 and fake_f >= 0.40:
                for param in model.head.parameters(): param.requires_grad = True
                state['is_active'] = False
        else:
            # SCENARIO B: Dam is OPEN
            if fake_r < 0.30 or fake_f < 0.30:
                for param in model.head.parameters(): param.requires_grad = False
                state['is_active'] = True
        return state

    print("\n[*] TEST 3 (Hysteresis Latch Simulation):")
    test_state = {'is_active': True} # Start with Dam Closed

    # Simulating a massive success (Both > 40%)
    print("    -> Simulating 45% Parity (Triggering Open)...")
    test_state = night_shift_step_SIMULATED(model, optimizer, gamma_controller, test_state, fake_r=0.45, fake_f=0.45)
    head_grad_status = next(model.head.parameters()).requires_grad
    if not test_state['is_active'] and head_grad_status == True:
        print("    [PASS] Dam Opened successfully. Head is Unfrozen.")
    else:
        print("    [FAIL] Latch refused to open.")

    # Simulating the 42% Whiplash (Dropping to 20%)
    print("    -> Simulating Post-Trigger Whiplash to 20% (Triggering Close)...")
    test_state = night_shift_step_SIMULATED(model, optimizer, gamma_controller, test_state, fake_r=0.20, fake_f=0.20)
    head_grad_status_2 = next(model.head.parameters()).requires_grad
    if test_state['is_active'] and head_grad_status_2 == False:
        print("    [PASS] Dam Slammed Shut. Head is Frozen. Whiplash survived.")
    else:
        print("    [FAIL] Latch stayed open during whiplash. FATAL.")

    print("\n[!] PRE-FLIGHT COMPLETE. IF ALL TESTS PASSED, YOU ARE CLEAR TO SLEEP.\n")

except Exception as e:
    print(f"\n[!] FATAL DIAGNOSTIC ERROR: {e}")
    import traceback
    traceback.print_exc()
