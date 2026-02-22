import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import math
import time
import os
import sys
import argparse
import json
import manifolds 
from torch.amp import autocast, GradScaler
from transformers import GPT2TokenizerFast
import pynvml
import threading
import traceback
from ghost_core import PrimalTRMCore, GhostLinear, AdaptiveVarianceTamer, GhostLinearTandem

# Forces the terminal output to handle whatever weirdness your model generates
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # Not a real TextIO — imported as a module (e.g. from overfit_test.py)

# ------------------------------------------------------------------------------
# 0. LOGGING INFRASTRUCTURE (Tee)
# ------------------------------------------------------------------------------
class LoggerTee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to persistent log for monitor
sys.stdout = LoggerTee("training_v5_71.log")

# Set global seed for perfect experiment parity
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)

# ------------------------------------------------------------------------------
# 1. ARGUMENT PARSING
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Antigravity Modular Trainer')
parser.add_argument('--manifold', type=str, default='unified', choices=['prime', 'linear', 'fp8', 'unified'], help='Manifold type')
parser.add_argument('--stats_file', type=str, default='stats_project_real.json', help='Stats output file')
parser.add_argument('--checkpoint_prefix', type=str, default='primal', help='Checkpoint prefix')
parser.add_argument('--dim', type=int, default=768, help='Model dimension (768 for 0.1B)')
parser.add_argument('--n_layers', type=int, default=12, help='Number of layers')
parser.add_argument('--n_heads', type=int, default=12, help='Number of heads')
parser.add_argument('--mode', type=str, default='primal', choices=['primal', 'standard'], help='Training mode: primal (discrete) or standard (FP16)')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
args = parser.parse_args()

# Protocol v2.0 'Pure Primal' Configuration
# CONFIG = {
#     'vocab_size': 50257, 
#     'seq_len': 1024,      
#     'dim': 768,      
#     'n_layers': 12,      
#     'n_heads': 12,       
#     'lr': 1e-4,          
#     'batch_size': 1,     
#     'grad_accum': 128,     
#     'scale_decay': 1e-4,
#     'stats_file': args.stats_file,
#     'prefix': args.checkpoint_prefix,
#     'mode': args.mode,
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu'
# }

CONFIG = {
    'vocab_size': 50257, 
    'seq_len': 1024,      
    'dim': 1024, # Expanded to Wide-Body Light (from 768)
    'iterations': 4, # Reduced depth for Wide-Body spec [Protocol v5.78]
    'n_layers': 12,      
    'n_heads': 12,       
    'lr': 1e-4,          
    'batch_size': 1,     
    'grad_accum': 128,     
    'scale_decay': 1e-4,
    'stats_file': args.stats_file,
    'prefix': args.checkpoint_prefix,
    'mode': args.mode,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ------------------------------------------------------------------------------
# 2. LUT GENERATION
# ------------------------------------------------------------------------------
if args.manifold == 'prime':
    PRIMAL_LUT = manifolds.generate_int16_prime_manifold(device=CONFIG['device'])
elif args.manifold == 'fp8':
    PRIMAL_LUT = manifolds.generate_fp8_manifold(device=CONFIG['device'])
elif args.manifold == 'unified':
    PRIMAL_LUT = manifolds.generate_int16_linear_manifold(device=CONFIG['device'])
else:
    PRIMAL_LUT = manifolds.generate_int16_linear_manifold(device=CONFIG['device'])

FINE_LUT = PRIMAL_LUT 

# ------------------------------------------------------------------------------
# 3. SYMMETRIC VOTING ENGINE
# ------------------------------------------------------------------------------
# (Control logic moved back to model modules)

# --- GAMMA CONTROLLER (Protocol v5.96) ---
class GammaController:
    def __init__(self, reasoning=1.0, head=0.0):
        self.ratios = {"reasoning": reasoning, "head": head}
    def set_redistribution_ratio(self, reasoning_weight, head_weight):
        self.ratios = {"reasoning": reasoning_weight, "head": head_weight}
    @property
    def head(self):
        return self.ratios.get("head", 0.0)
    @property
    def reasoning(self):
        return self.ratios.get("reasoning", 1.0)
    def update(self, core=None, head=None):
        if core is not None: self.ratios['reasoning'] = core
        if head is not None: self.ratios['head'] = head

# ------------------------------------------------------------------------------
# 4. THE TRAINING SENTINEL v5.6
# ------------------------------------------------------------------------------
class AntigravitySentinelV571:
    def __init__(self, model, optimizer, scaler=None, sensitivity=0.05, supervisor=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.sensitivity = sensitivity
        self.supervisor = supervisor # Link to supervisor for Protocol v6.10
        self.stall_streak = 0
        self.adaptive_prob = 0.001 
        self.is_paused = False

    @torch.no_grad()
    def apply_safestep_and_vote(self, current_loss_val=10.0, max_norm=1.0, current_step=None):
        if self.scaler: self.scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        total_flips = 0; num_params = 0; batch_max_stride = 0; stride_sums = 0.0; active_layers = 0
        current_lr = self.optimizer.param_groups[0]['lr']
        
        for name, module in self.model.named_modules():
            # 1. CORE & HEAD: Process the GhostLinearTandem layers
            if isinstance(module, GhostLinearTandem):
                if name == "head" and self.supervisor and getattr(self.supervisor, 'venting_active', False):
                    # [PROTOCOL v6.10: CONTROLLED VENTING]
                    # Override the normal flip cap with the 1% vent limit
                    vent_cap = int(module.base_idx.numel() * self.supervisor.max_vent_rate)
                    stats = module.apply_tandem_votes(learning_rate=current_lr, custom_cap=vent_cap)
                elif name == "head" and current_step is not None and current_step < 1000:
                    # [PROTOCOL v6.08: MASTER LOCK]
                    # Standard Lock: No flips allowed for the head until Step 1000 (unless venting)
                    stats = {"flips": 0, "avg_stride": 0, "max_stride": 0}
                else:
                    # Normal operation or reasoning_gate
                    stats = module.apply_tandem_votes(learning_rate=current_lr)
                
                if isinstance(stats, dict):
                    total_flips += stats["flips"]
                    batch_max_stride = max(batch_max_stride, stats["max_stride"])
                    if stats["flips"] > 0: 
                        stride_sums += stats["avg_stride"]
                        active_layers += 1
                num_params += module.base_idx.numel() + module.fine_idx.numel()
            
            # 2. FRONT DOOR FIX: Process the TrueShadowlessEmbedding layer
            elif isinstance(module, TrueShadowlessEmbedding):
                # Require 5 solid votes in the same direction before shifting a token's index
                emb_flips = module.apply_votes(consensus_threshold=5) 
                if isinstance(emb_flips, int):  # Ensure we get a valid integer back
                    total_flips += emb_flips
                    if emb_flips > 0:
                        active_layers += 1
                        stride_sums += 1.0  # Embeddings jump exactly 1 Prime coordinate
                        batch_max_stride = max(batch_max_stride, 16)
                # num_params counts both buffers
                num_params += module.base_idx.numel() + module.fine_idx.numel()

        if self.scaler: 
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else: 
            self.optimizer.step()
            
        flip_rate = (total_flips / max(1, num_params)) * 100
        avg_stride = (stride_sums / active_layers) if active_layers > 0 else 0.0
        return True, total_norm, flip_rate, avg_stride, batch_max_stride, current_lr

    def check_vitals(self, flip_rate, step, current_loss):
        """Heartbeat monitor (v5.71)."""
        if math.isnan(current_loss) or math.isinf(current_loss):
             print(f"[!] Critical Error: NaN Loss at Step {step}. Protection Active.")
        
        # Review Lock
        should_pause = False
        if current_loss <= 3.8 and not self.is_paused:
             print(f"\n[!] TARGET REACHED: Loss {current_loss:.4f} <= 3.8. Locking.")
             self.is_paused = True
             should_pause = True
        return should_pause

# --- VOTER PRECINCT MAPPER (Protocol v5.88/v5.93) ---
def map_voter_activity(model, block_size=None):
    """
    Scans the Master Voter precincts to see where the 'Grammar' vs 'Facts' 
    pressure is building.
    """
    report = []
    structured_data = []
    
    report.append("\n" + "="*40)
    report.append("      ANTIGRAVITY MANIFOLD VOTER MAP      ")
    report.append("="*40)
    
    for name, module in model.named_modules():
        if hasattr(module, 'vote_buffer'):
            bs = block_size if block_size is not None else getattr(module, 'block_size', 32)
            buffer = module.vote_buffer.data.abs()
            flat_buf = buffer.view(-1)
            num_blocks = flat_buf.numel() // bs
            blocks = flat_buf.view(num_blocks, bs)
            block_pressure = blocks.float().mean(dim=1)
            
            ready_count = (block_pressure >= 0.8).sum().item()
            activity_pct = (ready_count / num_blocks) * 100
            
            map_width = 40
            ascii_map = ""
            if num_blocks >= map_width:
                chunks = torch.chunk(block_pressure, map_width)
                for c in chunks:
                    val = c.mean().item()
                    if val > 0.9: ascii_map += "█"
                    elif val > 0.7: ascii_map += "▓"
                    elif val > 0.4: ascii_map += "▒"
                    else: ascii_map += "░"
            else:
                ascii_map = "░" * map_width
                
            report.append(f"\nLAYER: {name}")
            report.append(f"Total Voter Precincts: {num_blocks}")
            report.append(f"Active/Near-Consensus: {ready_count} ({activity_pct:.2f}%)")
            report.append(f"MAP: [{ascii_map}]")
            
            structured_data.append({
                "layer": name,
                "num_blocks": num_blocks,
                "active_count": ready_count,
                "active_pct": round(activity_pct, 2),
                "map": ascii_map
            })

    report.append("\n" + "="*40)
    report.append("LEGEND: █=Update Ready | ▓=High Pressure | ░=Static")
    report.append("="*40 + "\n")
    
    # Save structured data for Trinity Peak
    try:
        with open("voter_map_stats.json", "w") as f:
            json.dump(structured_data, f)
    except: pass
    
    output = "\n".join(report)
    print(output)
    
    # Return both the report and the peak activity percentage for the Supervisor
    max_pct = max([d['active_pct'] for d in structured_data]) if structured_data else 0.0
    return output, max_pct

# ==============================================================================
# PROTOCOL v5.96: GRADIENT DAM WITH HYSTERESIS LATCH
# ==============================================================================
# ==============================================================================
# PROTOCOL v6.00: NIGHT SHIFT SUPERVISOR
# ==============================================================================
def night_shift_step(model, optimizer, gamma_controller, dam_state, total_flips=0, active_layers=0, current_step=None):
    """Protocol v6.08: MASTER LOCK.
    
    Hold the dam closed until step 1000 to ensure core focus.
    Only trigger Emergency Dam Release if the ENTIRE model is truly stalled
    (zero flips across all layers for 5+ consecutive steps) AND step >= 1000.
    """
    # [PROTOCOL v6.08: MASTER LOCK]
    if current_step is not None and current_step < 1000:
        if not dam_state.get('is_active', False) or gamma_controller.head > 0:
             print(f"[NIGHT SHIFT] 🔒 MASTER LOCK active (Step {current_step}/1000). Forcing Dam CLOSED for core focus.")
             for p in model.head.parameters():
                 p.requires_grad = False
             gamma_controller.set_redistribution_ratio(1.0, 0.0)
             dam_state['is_active'] = True
             dam_state['manual_lock'] = True
        return dam_state
    # --- HALLUCINATION PATCH ---
    # Old code panicked if the reasoning_gate was at 0 flips, force-opening the dam.
    # New code: if refinement_gate or embeddings are flipping, we're alive.
    if active_layers > 0:
        # System is alive. Keep dam in its current state. No emergency.
        # Reset the stall counter since we have activity.
        dam_state['stall_streak'] = 0
        return dam_state
    
    # All layers are showing zero flips. Increment stall streak.
    stall_streak = dam_state.get('stall_streak', 0) + 1
    dam_state['stall_streak'] = stall_streak
    
    if stall_streak >= 5:
        # Only NOW do we declare a TRUE emergency and release the dam.
        print(f"[NIGHT SHIFT] 🚨 TRUE STARVATION: Zero flips for {stall_streak} consecutive steps. Releasing Dam.")
        for p in model.head.parameters():
            p.requires_grad = True
        gamma_controller.set_redistribution_ratio(0.60, 0.40)
        dam_state['is_active'] = False
        dam_state['manual_lock'] = False
        dam_state['stall_streak'] = 0  # Reset after action
    else:
        print(f"[NIGHT SHIFT] ⏳ Low-flip warning (streak {stall_streak}/5). Monitoring...")
    
    return dam_state

def break_manifold_stall(model, dam_state):
    """ --- EMERGENCY STALL BREAKER --- """
    print("\n[!] MANIFOLD STALL DETECTED. Adjusting Sensitivity...")
    
    # Save training state before jolt as requested
    torch.save(model.state_dict(), "stalled_state_before_jolt.pt")
    
    for module in model.modules():
        if isinstance(module, GhostLinearTandem):
            # 1. Lower the barrier to entry (Protocol v6.04)
            module.sensitivity *= 0.8  
            # 2. Lower the flip threshold
            module.supermajority_threshold = 12 
            # 3. Inject a 'Stochastic Jolt'
            if hasattr(module, 'inject_noise'):
                module.inject_noise(jitter_range=8)
            
    print("[*] Jolt complete. Barriers lowered. Watching for Flips...\n")
    return dam_state

def force_feed_core(model, gamma_controller, dam_state):
    """ EMERGENCY OVERRIDE: SLAMMING DAM SHUT & LOCKING IT """
    print(f"\n[!] STARVATION OVERRIDE ENGAGED.")
    print(f"    Action: Manual Lock Applied. Automation Suspended.")
    
    # 1. Engage the Manual Lock (The Fix)
    dam_state['manual_lock'] = True
    
    # 2. Force the State
    dam_state['is_active'] = True 
    for p in model.head.parameters(): 
        p.requires_grad = False
    
    # 3. Redirect Gamma
    gamma_controller.set_redistribution_ratio(1.0, 0.0)
    
    return dam_state

# --- PERFECTION AUTOPILOT (Protocol v5.90) ---
class AntigravityAutopilot:
    def __init__(self, model, initial_lr=2.0, patience=200):
        self.model = model
        self.base_lr = initial_lr
        self.patience = patience
        self.loss_history = []
        self.phase = 1  # Start at Block 32
        self.best_loss = float('inf')
        self.steps_since_improvement = 0
        self.is_suspended = False
        
    def suspend(self):
        self.is_suspended = True
        
    def resume(self):
        self.is_suspended = False
        
    def step(self, current_loss, current_step):
        if self.is_suspended:
            return True
        self.loss_history.append(current_loss)
        
        # Track Improvement
        if current_loss < self.best_loss - 0.005:
            self.best_loss = current_loss
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

        # PHASE 1 -> PHASE 2 TRANSITION (The High-Resolution Shift)
        # Condition: Loss is sub-6.0 OR we've stalled at Phase 1 for too long.
        if self.phase == 1 and (current_loss < 6.0 or self.steps_since_improvement > self.patience):
            print(f"\n[AUTOPILOT] CRITICAL EVENT: SHIFTING TO PHASE 2 (BLOCK SIZE 16)")
            self.phase = 2
            self.update_manifold_block_size(new_size=16)
            self.steps_since_improvement = 0 
            # We also drop the LR as we shift to higher resolution to avoid destabilization
            self.base_lr *= 0.8 

        # PHASE 2 -> PHASE 3 (The Asymptotic Freeze)
        if self.phase == 2 and (current_loss < 5.0 or self.steps_since_improvement > self.patience):
            print(f"\n[AUTOPILOT] CRITICAL EVENT: SHIFTING TO PHASE 3 (ASYMPTOTIC REFINEMENT)")
            self.phase = 3
            self.steps_since_improvement = 0
            self.base_lr *= 0.5

        # DYNAMIC LEARNING RATE MANAGEMENT
        if self.phase == 3:
            if self.steps_since_improvement > 10:
                self.base_lr *= 0.99
        
        # FINAL PERFECTION TRIGGER
        if current_loss < 4.5 and self.steps_since_improvement > 500:
            print(f"\n[AUTOPILOT] PERFECTION REACHED. SAVING FINAL MANIFOLD.")
            torch.save(self.model.state_dict(), "FINAL_PERFECT_MANIFOLD.pt")
            return False 

        return True 

    def update_manifold_block_size(self, new_size):
        """ Injects the new block size directly into the Ghost layers. """
        for module in self.model.modules():
            # Protocol v6.05: Use buffer fill for Torch compatibility
            if hasattr(module, 'block_size_buf'):
                module.block_size_buf.fill_(new_size)
            elif hasattr(module, 'block_size'):
                try:
                    module.block_size = new_size
                except AttributeError:
                    pass
        print(f"[MANIFOLD] All Voter Precincts re-drawn to size: {new_size}")

# ==============================================================================
# PROTOCOL v6.06: DYNAMIC SUPERMAJORITY CLUTCH (NightShiftSupervisor)
# ==============================================================================
class NightShiftSupervisor:
    def __init__(self, model, optimizer, gamma_controller, base_thresh=20, min_thresh=17, max_thresh=24):
        self.model = model
        self.optimizer = optimizer
        self.gamma_controller = gamma_controller
        self.current_threshold = base_thresh
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        
        self.loss_history = []
        self.starvation_patience = 0
        self.window_size = 100
        
        # [PROTOCOL v6.10: CONTROLLED VENTING]
        self.venting_active = False
        self.max_vent_rate = 0.01

        # [PROTOCOL v6.11: SOFT-LANDING RELEASE]
        self.landing_pause_steps = 0
        self.last_running_std = 0.0
        if hasattr(model, 'output_tamer'):
            self.last_running_std = model.output_tamer.running_std.item()

    def get_layer_activity(self, layer_name):
        """Calculates precise activity for a named layer."""
        for name, module in self.model.named_modules():
            if name == layer_name and hasattr(module, 'vote_buffer'):
                bs = getattr(module, 'block_size', 32)
                buffer = module.vote_buffer.data.abs()
                flat_buf = buffer.view(-1)
                num_blocks = flat_buf.numel() // bs
                if num_blocks > 0:
                    blocks = flat_buf.view(num_blocks, bs)
                    block_pressure = blocks.float().mean(dim=1)
                    ready_count = (block_pressure >= 0.8).sum().item()
                    return ready_count / num_blocks
        return 0.0

    def adjust_throttle(self, active_consensus_pct, current_step=None):
        """
        [PROTOCOL v6.09: CONSENSUS-DRIVEN LR DECAY]
        [PROTOCOL v6.10: VENTING SYNC]
        """
        if current_step is not None:
            # [DIAGNOSTIC: CORE TRIPWIRE]
            if current_step >= 1200:
                core_activity = self.get_layer_activity('trm_core.reasoning_gate')
                if core_activity > 0.001: # More than 0.1% activity
                    print(f"\n🚀 [TRIPWIRE] CORE IGNITION DETECTED at Step {current_step}!")
                    print(f"Core Activity: {core_activity*100:.4f}% | Gear: {self.current_threshold}")

            # [PROTOCOL v6.11: SOFT-LANDING RELEASE]
            if 1001 <= current_step <= 1100:
                # Disengage venting
                self.venting_active = False
                
                # Safeguard Verification
                if hasattr(self.model, 'output_tamer'):
                    current_variance_std = self.model.output_tamer.running_std.item()
                    if self.last_running_std > 0 and current_variance_std > self.last_running_std * 1.20:
                        print(f"[NIGHT SHIFT] ⚠️ VARIANCE SPIKE DETECTED ({current_variance_std:.2f} vs {self.last_running_std:.2f}). Pausing landing.")
                        self.landing_pause_steps = 5
                    self.last_running_std = current_variance_std

                if self.landing_pause_steps > 0:
                    self.landing_pause_steps -= 1
                    if current_step % 10 == 0:
                        print(f"[NIGHT SHIFT] ⏸️ LANDING PAUSED: {self.landing_pause_steps} steps remaining.")
                else:
                    # Start the Gamma bleed
                    progress = (current_step - 1000) / 100.0
                    new_head_gamma = 0.5 * progress
                    new_core_gamma = 1.0 - new_head_gamma
                    self.gamma_controller.update(core=new_core_gamma, head=new_head_gamma)
                    
                    if current_step % 10 == 0:
                        print(f"[NIGHT SHIFT] 🛬 SOFT LANDING: Step {current_step}/1100 | Gamma (C/H): {new_core_gamma:.2f}/{new_head_gamma:.2f}")

                # Pin the LR to 0.00005 [v6.11 Mandatory]
                new_lr = 0.00005
            elif current_step > 1100:
                # Full release complete
                self.venting_active = False
                self.gamma_controller.update(core=0.5, head=0.5)
                
                # Normal data-driven scaling resumes
                if active_consensus_pct < 1.0:
                    new_lr = 0.00001
                elif active_consensus_pct < 4.0:
                    new_lr = 0.00005
                else:
                    new_lr = 0.00010
            elif 750 <= current_step < 1000:
                # [PROTOCOL v6.10: CONTROLLED VENTING]
                if not self.venting_active:
                    print(f"[NIGHT SHIFT] 🌬️ VENTING ACTIVE (Step {current_step}/1000). Max Vent Rate: 1%.")
                self.venting_active = True
                new_lr = 0.00005
            else:
                self.venting_active = False
                # Standard baseline
                if active_consensus_pct < 1.0:
                    new_lr = 0.00001
                elif active_consensus_pct < 4.0:
                    new_lr = 0.00005
                else:
                    new_lr = 0.00010
        else:
            # Fallback for no step provided
            if active_consensus_pct < 1.0:
                new_lr = 0.00001
            elif active_consensus_pct < 4.0:
                new_lr = 0.00005
            else:
                new_lr = 0.00010
            
        old_lr = self.optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"[NIGHT SHIFT] ⚙️  THROTTLE ADJUSTED: Peak Consensus {active_consensus_pct:.2f}% -> LR: {new_lr}")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
    def evaluate_and_shift(self, current_loss):
        """Call this once per epoch or every N steps."""
        self.loss_history.append(current_loss)
        
        # Keep the history window tight
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            
        # We need a full window to make structural decisions
        if len(self.loss_history) < self.window_size:
            return self.current_threshold
            
        # Split window to calculate the mathematical trend
        older_avg = sum(self.loss_history[:50]) / 50.0
        recent_avg = sum(self.loss_history[-50:]) / 50.0
        
        # ---------------------------------------------------------
        # CONDITION 1: THE UPSHIFT (Avalanche / Instability Detected)
        # ---------------------------------------------------------
        if recent_avg > older_avg + 0.15:  
            new_thresh = min(self.max_thresh, self.current_threshold + 1)
            if new_thresh != self.current_threshold:
                print(f"[NIGHT SHIFT] ⚠️ INSTABILITY DETECTED. HARDENING Core Consensus: {self.current_threshold} -> {new_thresh}")
                self._update_model_thresholds(new_thresh)
                self.current_threshold = new_thresh
                self.loss_history.clear() # Reset memory after a shift
                self.starvation_patience = 0
                return self.current_threshold
                
        # ---------------------------------------------------------
        # CONDITION 2: THE DOWNSHIFT (Gradient Starvation Detected)
        # ---------------------------------------------------------
        # If the loss hasn't moved meaningfully in either direction...
        elif abs(recent_avg - older_avg) < 0.02:
            self.starvation_patience += 1
            
            # Require sustained starvation before relaxing the rules
            if self.starvation_patience >= 3: 
                new_thresh = max(self.min_thresh, self.current_threshold - 1)
                if new_thresh != self.current_threshold:
                    print(f"[NIGHT SHIFT] 📉 GRADIENT STARVATION. RELAXING Core Consensus: {self.current_threshold} -> {new_thresh}")
                    self._update_model_thresholds(new_thresh)
                    self.current_threshold = new_thresh
                    self.loss_history.clear()
                    self.starvation_patience = 0
        else:
            # The model is learning normally; reset patience
            self.starvation_patience = 0
            
        return self.current_threshold

    def _update_model_thresholds(self, new_threshold):
        """Broadcasts the new physical laws to the entire core."""
        for module in self.model.modules():
            if hasattr(module, 'supermajority_threshold'):
                module.supermajority_threshold = new_threshold

# 4. THERMAL SENTINEL
class ThermalVigil(threading.Thread):
    def __init__(self, sentinel, temp_limit=82):
        super().__init__(daemon=True)
        self.sentinel = sentinel
        self.temp_limit = temp_limit
        self.heat_stress = False
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.active = True
        except:
            print("[!] TRM Error: Could not initialize NVIDIA hardware monitoring.")
            self.active = False

    def run(self):
        if not self.active: return
        while self.active:
            try:
                # Query direct hardware diode
                temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, 0)
                fan = pynvml.nvmlDeviceGetFanSpeed(self.gpu_handle)
                pwr = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000 # Watts
                
                # Update the Heat Stress flag
                self.heat_stress = (temp >= self.temp_limit)
                
                if self.heat_stress:
                    print(f"\n[!] TRM WARNING: GPU at {temp}°C | Fan: {fan}% | Power: {pwr:.1f}W", flush=True)
                    # Force LR drop via sentinel if needed
                    for param_group in self.sentinel.optimizer.param_groups:
                        param_group['lr'] = min(param_group['lr'], 5e-5)
            except: pass
            time.sleep(2) # Low overhead polling

# Architecture classes are now imported from ghost_core.py


class GhostEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, lut, vote_buffer, sensitivity):
        # High-Speed Snap for sorted manifolds (Protocol v6.05 Optimized)
        with torch.no_grad():
            # torch.searchsorted finds the insertion index in O(log N)
            indices = torch.searchsorted(lut, weight)
            
            # Bound and check neighbors to find the absolute nearest neighbor
            idx_left = (indices - 1).clamp(0, lut.size(0) - 1)
            idx_right = indices.clamp(0, lut.size(0) - 1)
            
            dist_left = torch.abs(weight - lut[idx_left])
            dist_right = torch.abs(weight - lut[idx_right])
            
            final_indices = torch.where(dist_left < dist_right, idx_left, idx_right)
            quantized_weight = lut[final_indices]
            
        ctx.save_for_backward(weight, quantized_weight)
        ctx.vote_buffer = vote_buffer
        ctx.sensitivity = sensitivity
        return quantized_weight

    @staticmethod
    def backward(ctx, grad_output):
        weight, quantized_weight = ctx.saved_tensors
        vote_buffer = ctx.vote_buffer
        sensitivity = ctx.sensitivity
        
        # Calculate significance (like GhostQuantFunction)
        grad_abs = torch.abs(grad_output)
        g_mean = grad_abs.mean()
        g_std = grad_abs.std()
        threshold = g_mean + (sensitivity * g_std)
        
        # Inject votes into the buffer
        significant_mask = (grad_abs > threshold).to(torch.int16)
        direction = -torch.sign(grad_output).to(torch.int16)
        votes = direction * significant_mask
        
        # Only add votes where gradients exist (crucial for embeddings)
        nonzero_mask = (grad_output != 0).to(torch.int16)
        vote_buffer.add_(votes * nonzero_mask)
        
        # Pass the raw gradient back to the continuous weight (STE)
        return grad_output, None, None, None

class TrueShadowlessEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, lut, sensitivity=0.05):
        super().__init__()
        self.lut = lut; self.vocab_size = vocab_size; self.dim = dim; self.sensitivity = sensitivity
        import math
        bound = int(math.sqrt(6.0 / (vocab_size + dim)) * 32768)
        random_combined = torch.randint(max(0, 32768 - bound), min(65535, 32768 + bound), (vocab_size, dim), dtype=torch.int32)
        
        self.register_buffer('base_idx', torch.div(random_combined, 256, rounding_mode='floor').to(torch.uint8))
        self.register_buffer('fine_idx', (random_combined % 256).to(torch.uint8))
        self.register_buffer('vote_buffer', torch.zeros(vocab_size, dim, dtype=torch.int16))

    def forward(self, input_ids):
        combined_idx = (self.base_idx.long() * 256) + self.fine_idx.long()
        proxy_weight = self.lut[combined_idx].detach().requires_grad_(True)
        if proxy_weight.requires_grad:
            proxy_weight.register_hook(lambda grad: self._harvest_votes(grad))
        return F.embedding(input_ids, proxy_weight)

    def _harvest_votes(self, grad_weight):
        with torch.no_grad():
            grad_abs = torch.abs(grad_weight)
            threshold = grad_abs.mean() + (self.sensitivity * grad_abs.std())
            significant_mask = (grad_abs > threshold).to(torch.int16)
            direction = -torch.sign(grad_weight).to(torch.int16)
            self.vote_buffer.add_(direction * significant_mask)

    def apply_votes(self, consensus_threshold=5):
        with torch.no_grad():
            if self.vote_buffer.abs().max() < consensus_threshold:
                self.vote_buffer = (self.vote_buffer.float() * 0.95).to(torch.int16)
                return 0

            authorized_flips = self.vote_buffer.abs() >= consensus_threshold
            num_flips = authorized_flips.sum().item()
            if num_flips == 0: return 0

            combined_idx = (self.base_idx.to(torch.int32) * 256) + self.fine_idx.to(torch.int32)
            step_direction = torch.sign(self.vote_buffer[authorized_flips]).to(torch.int32)
            
            # Antigravity Stride Map
            distance_from_zero = torch.abs(combined_idx[authorized_flips] - 32768)
            stride_decay = distance_from_zero.float() / 32768.0 
            dynamic_stride = torch.clamp((16.0 * (1.0 - stride_decay)).to(torch.int32), min=1)
            
            combined_idx[authorized_flips] = torch.clamp(combined_idx[authorized_flips] - (step_direction * dynamic_stride), 0, 65535)
            self.base_idx.copy_(torch.div(combined_idx, 256, rounding_mode='floor').to(torch.uint8))
            self.fine_idx.copy_((combined_idx % 256).to(torch.uint8))
            
            self.vote_buffer[authorized_flips] = 0
            self.vote_buffer = (self.vote_buffer.float() * 0.95).to(torch.int16)
            return num_flips

# --- GHOST GPT ARCHITECTURE (v6.05 PRIME-SYNC) ---
class GhostGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        # [!] THE TRANSPLANT: Out with the continuous, in with the discrete.
        # Ensure your custom True Shadowless embedding class is defined above this!
        self.token_emb = TrueShadowlessEmbedding(c['vocab_size'], c['dim'], lut=PRIMAL_LUT)
        
        # Keeping position embedding standard for now, unless you've primed this too.
        self.pos_emb = nn.Embedding(2048, c['dim']) 
        
        # The deep core (currently soaking up that 100/0 Gamma pressure)
        self.trm_core = PrimalTRMCore(c['dim'], num_iterations=c.get('iterations', 4), lut=PRIMAL_LUT)
        self.ln_f = nn.LayerNorm(c['dim'])
        self.head = GhostLinearTandem(c['dim'], c['vocab_size'], lut=PRIMAL_LUT, sensitivity=0.05)
        # [Protocol v6.07: SHOCK ABSORBER] Guards the final output logits.
        # Wider envelope than the core tamer — allows normal token spread
        # but absorbs catastrophic spikes when the reasoning_gate first activates.
        self.output_tamer = AdaptiveVarianceTamer(base_threshold=6.0, max_threshold=16.0)

    def forward(self, idx, targets=None, annealing_factor=1.0, training_step=0):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + (self.pos_emb(pos) * 2.5)
        
        # Internal Reasoning Loop — tamer is applied inside _forward_impl each iteration
        x_refined = self.trm_core(x, lut=PRIMAL_LUT)
        
        y = self.ln_f(x_refined)
        
        # Raw 50,257-dim logits from the Ghost head
        logits = self.head(y, lut=PRIMAL_LUT)
        
        # [Protocol v6.07: SHOCK ABSORBER]
        # ONLY the output_tamer guards the final logits.
        # trm_core.tamer belongs inside the recursive hidden-state loop — NOT here.
        logits = self.output_tamer(logits)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= CONFIG['seq_len'] else idx[:, -CONFIG['seq_len']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ------------------------------------------------------------------------------
# 5. DATA
# ------------------------------------------------------------------------------
def get_loader():
    from datasets import load_dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    def iterate():
        token_buffer = []
        for sample in dataset:
            tokens = tokenizer.encode(sample['text'], max_length=CONFIG['seq_len'], truncation=True) + [tokenizer.eos_token_id]
            token_buffer.extend(tokens)
            while len(token_buffer) >= CONFIG['seq_len'] + 1:
                yield torch.tensor(token_buffer[:CONFIG['seq_len'] + 1], dtype=torch.long)
                token_buffer = token_buffer[CONFIG['seq_len']:]
    
    class IterDS(IterableDataset):
        def __iter__(self):
            return iterate()

    return DataLoader(IterDS(), batch_size=CONFIG['batch_size'], num_workers=0, pin_memory=True)

def export_stats(stats_list):
    with open(CONFIG['stats_file'], "w") as f:
        json.dump(stats_list, f)
def train(resume=False):
    # [Protocol v6.07: CLEAN START]
    if not resume:
        print(f"\n[*] FRESH START: Protocol v6.07 — GhostGPT + AdaptiveVarianceTamer")
        for f in [CONFIG['stats_file'], "samples_coder.json", "perplexity_coder.json"]:
            if os.path.exists(f): os.remove(f)
    else:
        print(f"\n[*] RESUMING from checkpoint...")
        
    model = GhostGPT(CONFIG).to(CONFIG['device'])
    
    base_params = [p for n, p in model.named_parameters() if '_idx' not in n]
    optimizer = torch.optim.AdamW(base_params, lr=CONFIG['lr'])
    
    scaler = GradScaler('cuda')
    sentinel = AntigravitySentinelV571(model, optimizer, scaler=scaler)
    
    # ── Checkpoint Loading ────────────────────────────────────────────────────
    start_step = 0
    checkpoint_path = f"{CONFIG['prefix']}_live.pt"
    if resume and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                start_step = checkpoint['step']
            else:
                model.load_state_dict(checkpoint, strict=False)
                print(f"[*] Loaded bare state_dict from {checkpoint_path}")
            print(f"[*] Resurrected from {checkpoint_path}")
        except Exception as e:
            print(f"[!] Checkpoint load failed: {e}. Starting fresh.")
    elif not resume and os.path.exists(checkpoint_path):
        print(f"[*] Ignoring stale checkpoint — fresh start requested.")
    
    stats_history = []

    # Thermal Sentinel (Shared via Governor concept)
    # thermal_sentry = ThermalVigil(governor)
    # thermal_sentry.start()
    
    print(f"[*] Starting Antigravity v6.07 — TinyStories Phase 2")
    
    loader = get_loader()
    
    t0 = time.time()
    base_clip_threshold = 1.0
    total_norm = torch.tensor(0.0, device=CONFIG['device'])
    first_pass = True

    model.train()
    t0 = time.time()
    
    # [STATS RESUME]
    stats_history = []
    if os.path.exists(CONFIG['stats_file']):
        try:
            with open(CONFIG['stats_file'], "r") as f:
                stats_history = json.load(f)
            if len(stats_history) > 0:
                start_step = stats_history[-1]['step'] + 1
                print(f"[*] Resuming stats from Step {start_step}")
        except:
            pass

    # --- GHOST CHECKPOINTER SETUP (Protocol v5.79) ---
    # We track descent milestones. Every time the loss breaks a floor, we save the manifold.
    milestones = [10.5, 10.0, 9.5, 9.0, 8.5, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0]
    next_milestone_idx = 0
    # import os # Redundant, already imported at top level
    os.makedirs("checkpoints", exist_ok=True)
    
    # Sentinel State
    base_clip_threshold = 1.0
    first_pass = True
    
    # Initialize Thermal Sentry
    thermal_sentry = ThermalVigil(sentinel, temp_limit=82)
    thermal_sentry.start()

    # Initialize Perfection Autopilot
    autopilot = AntigravityAutopilot(model)
    
    # ── Dam + Gamma — clean initialization ───────────────────────────────────
    # Dam starts CLOSED (Core Focus). manual_lock=False — Night Shift Supervisor
    # responds to real flip data, not hardcoded overrides.
    dam_state = {'is_active': True, 'step_count': 0, 'manual_lock': False, 'stall_streak': 0}
    gamma_controller = GammaController(reasoning=1.0, head=0.0)

    # Initialize Night Shift Supervisor (Dynamic Supermajority Clutch)
    supervisor = NightShiftSupervisor(model, optimizer, gamma_controller)

    # Freeze head gradient initially — 100% Gamma on reasoning core.
    for p in model.head.parameters():
        p.requires_grad = False

    print("[*] Night Shift Supervisor active.")
    print("[*] Dam CLOSED. Gamma: 100% Core / 0% Head. Head will unlock when core stabilises.\n")

    resonance_vent_active = False  # No forced vent on fresh start
    
    for i, batch in enumerate(loader):
        input_ids = batch[:, :-1].cuda(non_blocking=True)
        targets = batch[:, 1:].cuda(non_blocking=True)
        
        # Adjust step count for resume
        step = start_step + (i // CONFIG['grad_accum'])
        
        # [PROTOCOL v5.0] Check pause state
        while sentinel.is_paused:
            time.sleep(10)

        # [Protocol v5.92/v2.11 - TRS DISABLED]
        pass
            
        # Annealing
        annealing_factor = 0.5 + (0.5 * (min(step, 100) / 100.0))



        # 2. Mixed Precision Forward (v4.2 Passing Step)
        with autocast('cuda'):
            logits, loss = model(input_ids, targets=targets, annealing_factor=annealing_factor, training_step=step)
            loss = loss / CONFIG['grad_accum']
            
            # Scale Decay (v5.71: Unified Weight Penalty)
            loss_scale_penalty = 0
            for m in model.modules():
                if isinstance(m, GhostLinearTandem):
                    loss_scale_penalty = loss_scale_penalty + (m.scale - 1.0).pow(2).mean()
            total_loss = loss + (CONFIG['scale_decay'] * loss_scale_penalty)
            
            # 3. Entropy Watchdog (Diversity Check)
            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
                
                # [PROTOCOL v6.12: KINETIC JUMPSTART]
                # Synchronize the model's internal entropy buffer with the current result
                # This ensures the "Entropy Gate" in ghost_core.py can respond to real-time drift.
                model.trm_core.last_entropy.fill_(entropy)

        # Backward with Scalling
        scaler.scale(total_loss).backward()
        
        # [SENTINEL] Cooling Cycle (If Heat Stress Active)
        if thermal_sentry.heat_stress:
            time.sleep(0.05) 
        
        if (i + 1) % CONFIG['grad_accum'] == 0:
            
            # [NIGHT SHIFT v6.00: SENSOR SCAN]
            # Thermal Monitoring & Gamma collection still runs to feed the manifold metrics
            gamma_pool = 0.0
            max_variance = 0.0
            trs_ratios = gamma_controller.ratios
            dam_active = dam_state['is_active']
            
            # 1. GAMMA SIPHON & VARIANCE SCAN
            for name, module in model.named_modules():
                if hasattr(module, 'vote_buffer'):
                    var = torch.var(module.vote_buffer.float()).item()
                    max_variance = max(max_variance, var)
                    
                if "refinement_gate" in name and hasattr(module, 'siphon_gamma_energy'):
                    gamma_pool += module.siphon_gamma_energy(friction_reference=128)
            
            # [OVERFLOW PATCH v6.13] Hard ceiling on gamma_pool to prevent
            # absorb_gamma_energy from receiving an astronomically large total_pool.
            # --- LEGACY PRESERVATION (Pre-v6.13) ---
            # # At 2.88e9 the share per receiver overflowed int16 and stalled training.
            # gamma_pool = gamma_pool  # Uncapped
            # ----------------------------------------
            gamma_pool = min(gamma_pool, 500_000.0)

            # 2. GAMMA INJECTION (Protocol v5.94: Dynamic Split)
            if gamma_pool > 0:
                primary_pool = gamma_pool * trs_ratios['reasoning']
                secondary_pool = gamma_pool * trs_ratios['head']
                
                for name, module in model.named_modules():
                    if "reasoning_gate" in name and hasattr(module, 'absorb_gamma_energy'):
                        module.absorb_gamma_energy(primary_pool, friction_reference=128)
                    elif "head" in name and hasattr(module, 'absorb_gamma_energy'):
                        module.absorb_gamma_energy(secondary_pool, friction_reference=128)
                
                if step % 10 == 0:
                    dam_status = "ACTIVE (100/0)" if dam_active else "RELEASED (60/40)"
                    print(f"\n[GAMMA v5.94] Pool: {gamma_pool:.1f} | Dam: {dam_status} | Max Var: {max_variance:.1f}", flush=True)

            # [SENTINEL v5.73] Resolution & Vote engine (Step occurs here)
            # Must run BEFORE night_shift_step so we can pass live flip counts.
            success, total_norm, flip_rate, avg_stride, max_stride, current_lr = sentinel.apply_safestep_and_vote(current_loss_val=loss.item()*CONFIG['grad_accum'], max_norm=1.0, current_step=step)
            
            # Calculate active layers from sentinel data (non-zero flip_rate implies active layers)
            # We recalculate from sentinel's internal counters via flip_rate as a proxy
            _sentinel_active_layers = 1 if flip_rate > 0 else 0
            _sentinel_total_flips = int(flip_rate)
            
            # [PROTOCOL v6.07: HALLUCINATION PATCH]
            # Pass live sentinel data to night_shift_step so it makes a data-driven decision.
            dam_state = night_shift_step(model, optimizer, gamma_controller, dam_state, 
                                         total_flips=_sentinel_total_flips, 
                                         active_layers=_sentinel_active_layers,
                                         current_step=step)
            
            if success:
                optimizer.zero_grad()
                step += 1
                
                # [PROTOCOL v6.06: DYNAMIC SUPERMAJORITY CLUTCH]
                # The Night Shift Supervisor shifts the core's physical gears based on loss trends.
                current_gear = supervisor.evaluate_and_shift(loss.item() * CONFIG['grad_accum'])
                
                if step % 50 == 0:
                    supervisor_telemetry = {
                        "current_threshold": current_gear,
                        "min_thresh": supervisor.min_thresh,
                        "max_thresh": supervisor.max_thresh,
                        "starvation_patience": supervisor.starvation_patience
                    }
                    with open("supervisor_state.json", "w") as f:
                        json.dump(supervisor_telemetry, f)
                
                dt = time.time() - t0
                t0 = time.time()
                tps = (CONFIG['batch_size'] * CONFIG['seq_len'] * CONFIG['grad_accum']) / dt
                
                print(f"[{args.manifold.upper()}] Step {step} | Loss: {loss.item()*CONFIG['grad_accum']:.4f} | Entropy: {entropy:.2f} | Flips: {flip_rate:.4f}% | Stride: {avg_stride:.2f} (Max {max_stride}) | LR: {current_lr:.5f} | TPS: {tps:.2f}", flush=True)

                stats_history.append({
                    "step": step,
                    "loss": round(float(loss.item()*CONFIG['grad_accum']), 4),
                    "entropy": round(float(entropy), 2),
                    "flips": round(float(flip_rate), 4),
                    "avg_stride": round(float(avg_stride), 4),
                    "max_stride": int(max_stride),
                    "lr": round(float(current_lr), 6),
                    "norm": round(float(total_norm.item()), 4),
                    "tps": round(float(tps), 2),
                    "timestamp": time.time()
                })
                export_stats(stats_history)

                # Autopilot transition logic (Stripped of LR power)
                continue_training = autopilot.step(loss.item()*CONFIG['grad_accum'], step)
                if not continue_training:
                    print("[*] Autopilot signaled training completion.")
                    break
            # Transition safeguarding is handled by Night Shift's variance shield

            if step % 10 == 0 or first_pass:
                torch.save(model.state_dict(), f"{CONFIG['prefix']}_live.pt")
                
                # [WORD SALAD] Generation
                text = "N/A (Generation skipped)"
                if step % 50 == 0 or first_pass:
                    with torch.no_grad():
                        context = torch.tensor([[50256]], dtype=torch.long, device=CONFIG['device'])
                        generated = model.generate(context, max_new_tokens=64)
                        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
                        text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
                        print(f"\n[WORD SALAD] Step {step}: {text[:100]}...\n", flush=True)

                # [VOTER MAP] Snapshot & Save
                map_report, max_activity_pct = map_voter_activity(model)
                supervisor.adjust_throttle(max_activity_pct, current_step=step)
                with open("voter_map_snapshot.txt", "w", encoding='utf-8') as f:
                    f.write(f"--- ANTIGRAVITY MANIFOLD REPORT ---\n")
                    f.write(f"TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"TRAINING STEP: {step}\n")
                    f.write(f"\n[LATEST WORD SALAD]\n{text}\n")
                    f.write(map_report)
                
                # Append to samples file
                if step % 50 == 0 or first_pass:
                    sample_entry = {"step": step, "text": text, "timestamp": time.time()}
                    samples_path = "samples_coder.json"
                    existing_samples = []
                    if os.path.exists(samples_path):
                        try:
                            with open(samples_path, "r") as f:
                                existing_samples = json.load(f)
                        except: pass
                    existing_samples.append(sample_entry)
                    if len(existing_samples) > 10:
                        existing_samples = existing_samples[-10:]
                    with open(samples_path, "w") as f:
                        json.dump(existing_samples, f)

            if step % 100 == 0 or first_pass:
                # [PERPLEXITY] Monitor
                ppl = math.exp(loss.item() * CONFIG['grad_accum'])
                
                ppl_entry = {"step": step, "perplexity": ppl, "timestamp": time.time()}
                ppl_path = "perplexity_coder.json"
                
                existing_ppl = []
                if os.path.exists(ppl_path):
                    try:
                        with open(ppl_path, "r") as f:
                            existing_ppl = json.load(f)
                    except: pass
                
                existing_ppl.append(ppl_entry)
                # Keep last 100 points
                if len(existing_ppl) > 100:
                    existing_ppl = existing_ppl[-100:]
                    
                with open(ppl_path, "w") as f:
                    json.dump(existing_ppl, f)

            first_pass = False

if __name__ == "__main__":
    try:
        train(resume=args.resume)
    except Exception as e:
        print(f"\n[!!!] CRITICAL FAILURE IN MAIN: {e}")
        traceback.print_exc()
        sys.exit(1)
