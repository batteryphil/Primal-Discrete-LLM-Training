import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

# ==============================================================================
# PROTOCOL v6.07: RESPIRATION FIX - GRADIENT BOOSTER
# ==============================================================================
# Decouples forward-pass stability from backward-pass gradient magnitude.
# The forward pass is a pure identity - no numerical changes.
# The backward pass artificially amplifies the signal to overcome the 0.05 residual bottleneck.
class GradientBooster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, boost_factor):
        ctx.boost_factor = boost_factor
        return x  # Identity: no change to activations

    @staticmethod
    def backward(ctx, grad_output):
        # Amplify the backpropagated signal to overcome the 0.05 residual bottleneck
        # 0.05^4 = 0.00000625 vanishing gradient is corrected by 20x boost
        return grad_output * ctx.boost_factor, None

# ------------------------------------------------------------------------------
# 1. GHOST QUANTIZATION CORE
# ------------------------------------------------------------------------------
class GhostQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, scale, lut, vote_buffer, sensitivity):
        weights = lut[indices.long()] * scale
        # [RECURSION FIX] Only save tensors needed for gradient version tracking
        ctx.save_for_backward(indices, scale)
        # Attach others to ctx without version tracking
        ctx.lut = lut
        ctx.vote_buffer = vote_buffer
        ctx.sensitivity = sensitivity 
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        indices, scale = ctx.saved_tensors
        lut = ctx.lut
        vote_buffer = ctx.vote_buffer
        sensitivity = ctx.sensitivity
        
        # Gradient Centering
        grad_output = grad_output - grad_output.mean(dim=1, keepdim=True)
        
        # [PROTOCOL v6.16: CHUNKED BACKWARD]
        grad_scale = torch.zeros_like(scale)
        chunk_size = 4096
        out_features = indices.size(0)
        
        for i in range(0, out_features, chunk_size):
            end = min(i + chunk_size, out_features)
            idx_chunk = indices[i:end].long()
            g_out = grad_output[i:end]
            
            # --- SCALE GRADIENT ---
            weights_proxy = lut[idx_chunk]
            grad_scale[i:end] = (g_out * weights_proxy).sum(dim=1, keepdim=True)
            
            # --- WHISPER VOTING ---
            direction = -torch.sign(g_out).to(torch.int16)
            grad_abs = torch.abs(g_out)
            
            # Per-chunk threshold (approximation of global entropy)
            threshold = grad_abs.mean() + (sensitivity * grad_abs.std())
            significant_mask = (grad_abs > threshold).to(torch.int16)
            
            vote_buffer[i:end].add_(direction * significant_mask)
            
        return None, grad_scale, None, None, None

class GhostLinear(nn.Module):
    def __init__(self, in_features, out_features, lut=None, sensitivity=0.05):
        super().__init__()
        self.sensitivity = sensitivity 
        self.lut = lut # Can be passed or set globally
        
        # Protocol v5.2: Hard-Integer Data Types
        self.register_buffer('grid_indices', torch.zeros(out_features, in_features, dtype=torch.int16))
        self.register_buffer('vote_buffer', torch.zeros(out_features, in_features, dtype=torch.int16))
        self.register_buffer('cooldown', torch.zeros(out_features, in_features, dtype=torch.int16))
        self.scale = nn.Parameter(torch.ones(out_features, 1))
        
        if lut is not None:
            # Initialize with random weights quantized to the LUT
            raw_w = torch.randn(out_features, in_features) * 0.02
            self.grid_indices.copy_(self.quantize_to_indices(raw_w, lut))

    def quantize_to_indices(self, weights, lut):
        # weights normalized by scale
        # [v5.0] Optimized for Linear Manifold (O(N) memory)
        # Assuming lut is a linear space between -1.0 and 1.0
        # If the LUT is not linear, we'd need another method, but Protocol v5.0 specifies Linear.
        with torch.no_grad():
            # Map [-1, 1] to [0, 65535]
            normalized = (weights + 1.0) / 2.0
            indices = torch.clamp((normalized * 65535).round(), 0, 65535).to(torch.int16)
        return indices

    def forward(self, x, lut=None, annealing_factor=1.0):
        target_lut = lut if lut is not None else self.lut
        if target_lut is None:
            raise ValueError("GhostLinear expects a LUT (Look-Up Table) for forward pass.")
            
        effective_sensitivity = self.sensitivity * annealing_factor
        weights = GhostQuantFunction.apply(
            self.grid_indices, self.scale, target_lut, self.vote_buffer, effective_sensitivity
        )
        return F.linear(x, weights)

    def apply_votes(self, adaptive_prob, name="Unknown", cooldown_steps=100):
        with torch.no_grad():
            mask = self.cooldown > 0
            self.cooldown[mask] -= 1
            self.scale.data = self.scale.data * 0.999 + (1.0 * 0.001)

        if self.vote_buffer.abs().max() == 0:
            return 0
            
        prob_tensor = torch.full_like(self.vote_buffer.float(), adaptive_prob)
        # [v5.0] Scaled Pressure Thresholds for int16
        high_pressure_mask = self.vote_buffer.abs() >= 128
        prob_tensor[high_pressure_mask] = 0.1 
        
        saturated_mask = self.vote_buffer.abs() >= 64
        prob_tensor[saturated_mask] = torch.clamp(prob_tensor[saturated_mask] * 2, max=0.5)
        
        success = torch.rand_like(prob_tensor) < prob_tensor
        valid_flips = success & (self.vote_buffer != 0) & (self.cooldown == 0)
        
        num_requested = torch.count_nonzero(valid_flips).item()
        is_head = self.grid_indices.numel() > 5000000 
        if is_head:
            max_allowed = max(1, int(self.grid_indices.numel() * 0.00005))
        else:
            max_allowed = max(1, int(self.grid_indices.numel() * 0.0005))

        if num_requested > max_allowed:
            abs_votes = self.vote_buffer.abs().float()
            abs_votes[~valid_flips] = -1.0
            flat_votes = abs_votes.view(-1)
            vals, _ = torch.topk(flat_votes, max_allowed)
            min_val = vals[-1]
            valid_flips = (abs_votes >= min_val) & (min_val > 0)

        flip_mask = valid_flips.to(torch.int16)
        update = self.vote_buffer.sign().to(torch.int16) * flip_mask
        new_indices = self.grid_indices.to(torch.int32) + update.to(torch.int32)
        # Protocol v5.0: Clamp to 16-bit range (0 to 65535)
        self.grid_indices.copy_(torch.clamp(new_indices, 0, 65535).to(torch.int16))
        
        if valid_flips.any():
            self.cooldown[valid_flips] = cooldown_steps
        
        self.vote_buffer[valid_flips] = 0
        num_flips = torch.count_nonzero(valid_flips).item()
        
        if num_flips > 1000:
             print(f"[!] AVALANCHE in {name} | Flips: {num_flips}", flush=True)

        friction = getattr(self, 'friction_penalty', 1)
        with torch.no_grad():
            pos_mask = self.vote_buffer > 0
            neg_mask = self.vote_buffer < 0
            self.vote_buffer[pos_mask] = torch.clamp(self.vote_buffer[pos_mask] - friction, min=0)
            self.vote_buffer[neg_mask] = torch.clamp(self.vote_buffer[neg_mask] + friction, max=0)
            
        return num_flips

    def inject_noise(self, jitter_range=2):
        """
        Protocol v5.2: Stochastic Breakthrough (Hard Integer).
        Adds random integer jitter to the vote buffer.
        """
        with torch.no_grad():
            # [v5.2] Pure Integer Noise
            noise = torch.randint(-jitter_range, jitter_range + 1, self.vote_buffer.shape, 
                                  device=self.vote_buffer.device, dtype=torch.int16)
            self.vote_buffer.add_(noise)

class FineGhostLinear(GhostLinear):
    def __init__(self, in_features, out_features, lut=None, sensitivity=0.8):
        super().__init__(in_features, out_features, lut=lut, sensitivity=sensitivity)

class StandardLinear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features)
    def forward(self, x, *args, **kwargs):
        return super().forward(x)
    def apply_votes(self, *args, **kwargs):
        return 0

# ------------------------------------------------------------------------------
# 2. SENTINEL STABILITY TOOLS
# ------------------------------------------------------------------------------
# [Protocol v6.07: SHOCK ABSORBER — Adaptive Logit Tamer]
# Replaces the static LogitTamer. Acts as an automatic transmission for Softmax
# temperature. Tracks running logit std-dev and only engages when a spike is
# detected — elastic absorption vs. a hard brick-wall clamp.
class AdaptiveVarianceTamer(nn.Module):
    def __init__(self, base_threshold=4.0, max_threshold=12.0, momentum=0.99):
        super().__init__()
        self.base_threshold = base_threshold
        self.max_threshold = max_threshold
        self.momentum = momentum
        # Running stats tracked as buffers so they survive checkpoint saves
        self.register_buffer('running_std', torch.tensor(base_threshold))
        self.register_buffer('temperature', torch.tensor(1.0))

    def forward(self, logits):
        if not self.training:
            # Inference: soft-divide by learned temperature
            return logits / self.temperature

        with torch.no_grad():
            current_std = logits.std()

            # Smoothly update the running standard deviation baseline
            self.running_std.mul_(self.momentum).add_(current_std * (1.0 - self.momentum))

            # Shock absorber threshold: 1.5x the running baseline, clamped to [base, max]
            safe_limit = max(self.running_std.item() * 1.5, self.base_threshold)
            safe_limit = min(safe_limit, self.max_threshold)

            if current_std > safe_limit:
                # Spike detected — scale logits down to the safe envelope
                scale = safe_limit / (current_std + 1e-6)
                logits = logits * scale

        return logits

# ------------------------------------------------------------------------------
# 3. RECURSIVE-REFINEMENT CORE (v4.0)
# ------------------------------------------------------------------------------
class PrimalTRMCore(nn.Module):
    # def __init__(self, dim, num_iterations=8, lut=None):
    #     super().__init__()
    #     self.iterations = num_iterations
    #     # [v5.71] Unified 16-bit Core
    #     self.reasoning_gate = GhostLinearTandem(dim * 3, dim, lut=lut, sensitivity=0.05)
    #     self.refinement_gate = GhostLinearTandem(dim, dim, lut=lut, sensitivity=0.05)
    #     
    #     # [v4.4 DYNAMIC BUFFERS -> v5.0]
    #     self.register_buffer('voltage_boost', torch.tensor(1.0)) # Warm start
    #     self.register_buffer('tamer_threshold', torch.tensor(8.0))
    #     self.register_buffer('current_step', torch.tensor(0))
    #     self.register_buffer('last_entropy', torch.tensor(11.0))
    #     
    #     # [v5.1] Scaling Support
    #     self.use_checkpointing = False
    #     self.tamer = LogitTamer(threshold=4.0) # Tightened for v5.6

    def __init__(self, dim, num_iterations=4, lut=None, vocab_size=32000, heads=32, mlp_dim=8192):
        super().__init__()
        self.iterations = num_iterations
        # Protocol v5.78 Wide-Body Params
        self.dim = dim
        self.heads = heads
        
        # [v5.71] Unified 16-bit Core (Preserved)
        self.reasoning_gate = GhostLinearTandem(dim * 3, dim, lut=lut, sensitivity=0.05)
        self.refinement_gate = GhostLinearTandem(dim, dim, lut=lut, sensitivity=0.05)
        
        # [v4.4 DYNAMIC BUFFERS -> v5.0]
        self.register_buffer('voltage_boost', torch.tensor(1.0))
        self.register_buffer('tamer_threshold', torch.tensor(8.0))
        self.register_buffer('current_step', torch.tensor(0))
        self.register_buffer('last_entropy', torch.tensor(11.0))
        
        # [v5.1] Scaling Support
        self.use_checkpointing = False
        # [Protocol v6.07] Replaced static LogitTamer with elastic Shock Absorber
        self.tamer = AdaptiveVarianceTamer(base_threshold=4.0, max_threshold=10.0)

    def update_friction(self, step, current_entropy):
        """
        Protocol v4.4: Semantic Smoothing.
        Decays voltage and tightens taming proportionally to manifold crystallization.
        """
        if current_entropy < 7.0:
            # Linear Decay of Voltage (3.0 -> 2.2 over 5,000 steps)
            if not hasattr(self, 'v44_start_step'):
                self.v44_start_step = step
            
            elapsed = step - self.v44_start_step
            decay_factor = min(1.0, elapsed / 5000.0)
            
            self.voltage_boost.fill_(3.0 - (0.8 * decay_factor))
            
            # Tamer Tightening (12.0 -> 9.0)
            # max(9.0, 12.0 - (steps * 0.0006))
            new_tamer = max(9.0, 12.0 - (elapsed * 0.0006))
            self.tamer_threshold.fill_(new_tamer)
            self.tamer.threshold = new_tamer
    
    def forward(self, x, lut=None):
        with torch.no_grad():
             # Placeholder for step increment if not provided by train loop
             # But train loop should probably set it. 
             # For now, we'll increment internally as a fallback.
             self.current_step.add_(1)
             
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, lut, use_reentrant=False)
        return self._forward_impl(x, lut)

    # def _forward_impl(self, x, lut=None):
    #     # x is the static anchor
    #     # Ensure x is 3D [batch, seq, dim]
    #     if x.dim() == 2:
    #         x = x.unsqueeze(0)
    #         
    #     y = torch.zeros_like(x) # Initial Answer
    #     z = torch.zeros_like(x) # Initial Workspace
    #
    #     # --- v4.2 CHECKPOINT: LINEAR ANCHOR ---
    #     combined = torch.cat([x, y, z], dim=-1)
    #     z_update = self.reasoning_gate(combined, lut=lut)
    #     z = torch.tanh(z_update)
    #     y_delta = self.refinement_gate(z, lut=lut)
    #     # Protocol v5.6: Stabilizing Refinement Gate (Initial damping)
    #     y = y + (y_delta * 0.25)
    #     
    #     if self.last_entropy < 10.0 and self.current_step >= 100:
    #         for t in range(1, self.iterations):
    #             combined = torch.cat([x, y, z], dim=-1)
    #             z_update = self.reasoning_gate(combined, lut=lut)
    #             z = torch.tanh(z_update) 
    #             
    #             y_delta = self.refinement_gate(z, lut=lut)
    #             # Protocol v5.6: Strict Dampened Recursive Refinement
    #             y = y + (y_delta * 0.05) 
    #             
    #             y = self.tamer(y)
    #         
    #     return y

    def ignite_recursive_core(self):
        """
        [PROTOCOL v6.12: KINETIC JUMPSTART]
        Signals that the core is engaging its internal reasoning loops.
        """
        pass

    def _forward_impl(self, x, lut=None):
        # x is the static anchor
        # Ensure x is 3D [batch, seq, dim]
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        y = torch.zeros_like(x) # Initial Answer
        z = torch.zeros_like(x) # Initial Workspace

        # --- v4.2 CHECKPOINT: LINEAR ANCHOR ---
        combined = torch.cat([x, y, z], dim=-1)
        z_update = self.reasoning_gate(combined, lut=lut)
        z = torch.tanh(z_update)
        y_delta = self.refinement_gate(z, lut=lut)
        # Protocol v5.6: Stabilizing Refinement Gate (Initial damping)
        y = y + (y_delta * 0.25)
        
        # [PROTOCOL v6.12: KINETIC JUMPSTART]
        # Lowering the gate from 10.0 to 10.80 to break the Ignition Lock
        # --- LEGACY PRESERVATION (Pre-v6.12) ---
        # if self.last_entropy < 10.0 and self.current_step >= 100:
        # ----------------------------------------
        if self.last_entropy < 10.80 and self.current_step >= 100:
            if not hasattr(self, '_ignition_logged'):
                self.ignite_recursive_core()
                print(f"🚀 [IGNITION] Core logic engaged at Entropy: {self.last_entropy:.4f}")
                self._ignition_logged = True
            for t in range(1, self.iterations):
                combined = torch.cat([x, y, z], dim=-1)
                
                # --- RESPIRATION FIX (Protocol v6.07) ---
                # Multiply gradients flowing backward by 20.0 to exactly cancel
                # the 0.05 residual bottleneck (0.05 * 20.0 = 1.0 effective gradient scale).
                # The forward pass is pure identity — numerical values are unchanged.
                boosted_combined = GradientBooster.apply(combined, 20.0)
                
                z_update = self.reasoning_gate(boosted_combined, lut=lut)
                z = torch.tanh(z_update)
                # -----------------------------------------
                
                y_delta = self.refinement_gate(z, lut=lut)
                # Protocol v5.6: Strict Dampened Recursive Refinement
                y = y + (y_delta * 0.05) 
                
                # --- ACTIVATION TAMER (ANTI-RESONANCE CLAMP) [Protocol v5.78] ---
                y = torch.clamp(y, -10.0, 10.0)

                y = self.tamer(y)
            
        return y
# ------------------------------------------------------------------------------
# 4. CODER'S TANDEM (Protocol v5.6)
# ------------------------------------------------------------------------------
class GhostTandemQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base_idx, fine_idx, scale, lut, vote_buffer, sensitivity):
        try:
            import primal_cuda
            if base_idx.is_cuda and scale.is_cuda and lut.is_cuda:
                weights = primal_cuda.forward(base_idx, fine_idx, scale, lut)
            else:
                combined_idx = (base_idx.to(torch.int32) * 256 + fine_idx.to(torch.int32)).long()
                weights = lut[combined_idx] * scale
        except ImportError:
            combined_idx = (base_idx.to(torch.int32) * 256 + fine_idx.to(torch.int32)).long()
            weights = lut[combined_idx] * scale
            
        ctx.save_for_backward(base_idx, fine_idx, scale)
        ctx.lut = lut
        ctx.vote_buffer = vote_buffer
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        base_idx, fine_idx, scale = ctx.saved_tensors
        lut = ctx.lut
        vote_buffer = ctx.vote_buffer
        
        # Calculate max for normalization across the WHOLE gradient
        grad_max = grad_output.abs().max().item() + 1e-12
        
        try:
            import primal_cuda
            if grad_output.is_cuda and base_idx.is_cuda:
                grad_scale = primal_cuda.backward(
                    grad_output, base_idx, fine_idx, scale, lut, vote_buffer, float(grad_max)
                )
                return None, None, grad_scale, None, None, None
        except ImportError:
            pass

        # [Fallback PROTOCOL v6.16: CHUNKED BACKWARD]
        grad_scale = torch.zeros_like(scale)
        chunk_size = 4096 
        out_features = base_idx.size(0)
        
        for i in range(0, out_features, chunk_size):
            end = min(i + chunk_size, out_features)
            
            # --- SCALE GRADIENT ---
            b_chunk = base_idx[i:end].to(torch.int32)
            f_chunk = fine_idx[i:end].to(torch.int32)
            c_idx = (b_chunk * 256 + f_chunk).long()
            
            g_out = grad_output[i:end]
            grad_scale[i:end] = (g_out * lut[c_idx]).sum(dim=1, keepdim=True)
            
            # --- VOTE INJECTION ---
            g_normed = g_out / grad_max
            pressure = (g_normed * 100.0 * scale[i:end].sign()).to(torch.int32)
            
            # Update vote buffer in-place
            current_votes = vote_buffer[i:end].to(torch.int32)
            new_votes = torch.clamp(current_votes + pressure, -32760, 32760)
            vote_buffer[i:end].copy_(new_votes.to(torch.int16))
            
        return None, None, grad_scale, None, None, None

class GhostLinearTandem(nn.Module):
    def __init__(self, in_features, out_features, lut=None, sensitivity=0.09):
        super().__init__()
        self.sensitivity = sensitivity
        self.lut = lut
        
        # --- INTEGER-NATIVE XAVIER INITIALIZATION ---
        # Calculate standard deviation for weights to break symmetry
        import math
        bound = math.sqrt(6.0 / (in_features + out_features))
        
        # Map float bounds to our 16-bit manifold (0 to 65535, center is 32768)
        index_bound = int(bound * 32768)
        
        print(f"[GHOST] Init {out_features}x{in_features} ghost layer...")
        
        # Generate random uniform integer indices centered around zero (32768)
        random_combined = torch.randint(
            max(0, 32768 - index_bound), 
            min(65535, 32768 + index_bound), 
            (out_features, in_features), 
            dtype=torch.int32
        )
        
        # Split into Base (MSB) and Fine (LSB) uint8 buffers
        self.register_buffer('base_idx', torch.div(random_combined, 256, rounding_mode='floor').to(torch.uint8))
        self.register_buffer('fine_idx', (random_combined % 256).to(torch.uint8))
        
        # --- BUFFER INITIALIZATION ---
        self.register_buffer('vote_buffer', torch.zeros(out_features, in_features, dtype=torch.int16))
        self.scale = nn.Parameter(torch.ones(out_features, 1))
        
        # Protocol v5.98: Settings
        self.supermajority_threshold = 20 
        
    @property
    def block_size(self):
        # Default block size, no longer a buffer
        return 32
        
    @block_size.setter
    def block_size(self, value):
        self.block_size_buf.fill_(value)

    def forward(self, x, lut=None, annealing_factor=1.0):
        target_lut = lut if lut is not None else self.lut
        if target_lut is None:
            raise ValueError("GhostLinearTandem expects a LUT.")
            
        weights = GhostTandemQuantFunction.apply(
            self.base_idx, self.fine_idx, self.scale, 
            target_lut, self.vote_buffer, self.sensitivity * annealing_factor
        )
        return F.linear(x, weights)

    # def apply_tandem_votes(self, learning_rate=1.0):
    #     with torch.no_grad():
    #         # 1. CALCULATE CONTINUOUS PRESSURE
    #         pressure = self.vote_buffer.float() * learning_rate / 10.0
    #         
    #         # 2. THE VOTING MACHINE (Stochastic Rounding)
    #         base_steps = pressure.trunc() 
    #         remainder = (pressure - base_steps).abs()
    #         
    #         # ... (Rest of old logic) ...
    #         return {
    #             "flips": valid_moves.sum().item(), 
    #             "avg_stride": avg_stride, 
    #             "max_stride": max_stride
    #         }

    def apply_tandem_votes(self, learning_rate=1.0, block_size=None, custom_cap=None):
        if block_size is None: block_size = 32
        with torch.no_grad():
            flat_buffer = self.vote_buffer.view(-1)
            num_elements = flat_buffer.numel()
            flat_view = flat_buffer if num_elements % block_size == 0 else flat_buffer[:(num_elements // block_size) * block_size]
            num_blocks = flat_view.numel() // block_size
            blocks = flat_view.float().view(num_blocks, block_size)
            
            master_vote_magnitude = blocks.abs().mean(dim=1, keepdim=True)
            
            # [PROTOCOL v6.14: LR-SENSITIVE VOTING]
            # Multiply pressure by normalized LR (1.0 at baseline 1e-4)
            pressure_scale = learning_rate / 0.0001
            authorized_blocks = (master_vote_magnitude * pressure_scale >= self.supermajority_threshold)
            
            # [PROTOCOL v6.10: CONTROLLED VENTING]
            # If a custom_cap is provided (max allowed flipped parameters), 
            # we must restrict authorized_blocks to fit within that budget.
            if custom_cap is not None:
                max_blocks = max(1, int(custom_cap // block_size))
                num_authorized = authorized_blocks.sum().item()
                if num_authorized > max_blocks:
                    # Keep only the highest pressure blocks
                    vals, _ = torch.topk(master_vote_magnitude.view(-1), max_blocks)
                    min_val = vals[-1]
                    authorized_blocks = (master_vote_magnitude >= min_val) & (min_val > 0)

            step_direction = torch.sign(blocks) 
            
            combined_flat = (self.base_idx.view(-1)[:flat_view.numel()].to(torch.int32) * 256) + self.fine_idx.view(-1)[:flat_view.numel()].to(torch.int32)
            combined_blocks = combined_flat.view(num_blocks, block_size)
            
            # Core Stride Map
            block_center_distance = torch.abs(combined_blocks - 32768).float().mean(dim=1, keepdim=True)
            stride_decay = block_center_distance / 32768.0 
            dynamic_stride = torch.clamp((8.0 * (1.0 - stride_decay)).to(torch.long), min=1)
            
            update = (authorized_blocks.float() * step_direction * dynamic_stride).to(torch.int32)
            total_flips = authorized_blocks.sum().item() * block_size
            
            new_combined = torch.clamp(combined_flat - update.view(-1), 0, 65535)
            self.base_idx.view(-1)[:flat_view.numel()].copy_(torch.div(new_combined, 256, rounding_mode='floor').to(torch.uint8))
            self.fine_idx.view(-1)[:flat_view.numel()].copy_((new_combined % 256).to(torch.uint8))
            
            self.vote_buffer.view(-1)[:flat_view.numel()].view(num_blocks, block_size).masked_fill_(authorized_blocks, 0)
            self.vote_buffer = (self.vote_buffer.float() * 0.95).to(torch.int16)

            return {"flips": total_flips, "avg_stride": dynamic_stride.float().mean().item() if total_flips > 0 else 0.0, "max_stride": dynamic_stride.max().item() if total_flips > 0 else 0}

    def apply_radioactive_subsidy(self, friction_reference=128):
        """
        Protocol v5.92: TARGETED RADIOACTIVE SUBSIDY (TRS) Kernel Implementation
        [Adapted from CUDA Kernel Logic to PyTorch 16-bit Manifold]
        
        Logic:
        1. IDENTIFY DONORS: Siphon (Vote - Friction) * 0.80 from Active zones.
        2. BROADCAST: Sum the siphoned pool and divide by TOTAL PRECINCTS (not just receivers).
        3. APPLY SUBSIDY: Add to Qualifiers (0 < Abs(Vote) < 15% Friction).
        4. GOVERNOR: Cap resulting vote magnitude at 40% Friction.
        """
        with torch.no_grad():
            # 1. IDENTIFY DONORS & SIPHON (The "Tax")
            # We operate on absolute magnitude to preserve manifold symmetry
            abs_votes = self.vote_buffer.abs()
            active_mask = abs_votes >= friction_reference
            
            # Calculate excess: (Current - Threshold)
            excess = (abs_votes[active_mask].float() - friction_reference).clamp(min=0)
            
            # Entropy Loss: 0.20 (Recycle 80%)
            # shared_pool = excess * (1.0 - 0.20)
            siphoned_energy = excess.sum()
            subsidy_pool = siphoned_energy * 0.80
            
            # Clamp donors back to max friction
            # Apply sign to preserve direction
            self.vote_buffer[active_mask] = self.vote_buffer[active_mask].sign() * friction_reference

            # 2. BROADCAST (The "Targeted Welfare")
            # "new_total = current_votes + (layer_energy / 65536)"
            # This implies the pool is distributed across the entire layer topology,
            # but only picked up by the qualifiers?
            # Or physically added to everyone? The kernel says:
            # "if (current_votes > 0 && current_votes < qualification_ceiling) { ... }"
            # So the division is by total count, but the addition is selective.
            total_precincts = self.vote_buffer.numel()
            if total_precincts == 0: return {}
            
            universal_share = subsidy_pool / total_precincts
            
            # 3. APPLY SUBSIDY TO QUALIFIERS
            # Constraint: > 0 and < 15% of friction
            qualification_ceiling = friction_reference * 0.15
            
            # We strictly follow "current_votes > 0" magnitude logic
            qualifier_mask = (abs_votes > 0) & (abs_votes < qualification_ceiling)
            
            num_receivers = qualifier_mask.sum().item()
            
            if num_receivers > 0 and universal_share > 0:
                # Get qualifiers
                receivers = self.vote_buffer[qualifier_mask]
                receiver_signs = receivers.sign()
                receiver_mags = receivers.abs().float()
                
                # Add the universal share
                new_mags = receiver_mags + universal_share
                
                # 4. THE 40% GOVERNOR
                # "if (new_total > subsidy_cap) new_total = subsidy_cap;"
                subsidy_cap = friction_reference * 0.40
                new_mags = torch.min(new_mags, torch.tensor(subsidy_cap, device=self.vote_buffer.device))
                
                # Write back
                self.vote_buffer[qualifier_mask] = receiver_signs * new_mags.to(torch.int16)
                
            # 5. TELEMETRY
            final_variance = torch.var(self.vote_buffer.float()).item()
            final_active = (self.vote_buffer.abs() >= friction_reference).sum().item()
            final_consensus = (final_active / total_precincts) * 100.0
            
            return {
                "siphoned_energy": siphoned_energy.item(),
                "subsidy_pool": subsidy_pool.item(),
                "receivers": num_receivers,
                "share_per_receiver": universal_share.item(), # This is the broadcast amount per unit
                "universal_share": universal_share.item(),
                "delta_consensus": 0.0, # Placeholder or calc diff if we had initial
                "layer_variance": final_variance,
                "consensus_pct": final_consensus
            }

    def siphon_gamma_energy(self, friction_reference=128):
        """
        Protocol v5.93: REFINEMENT DAMPING (The Donor Kernel)
        Extracts overflow energy from saturated layers to populate the Gamma Pool.
        """
        with torch.no_grad():
            abs_votes = self.vote_buffer.abs()
            active_mask = abs_votes >= friction_reference
            
            # Siphon (Current - Threshold)
            excess = (abs_votes[active_mask].float() - friction_reference).clamp(min=0)
            
            # Gamma Energy = 80% of entropy-corrected excess
            gamma_energy = excess.sum() * 0.80
            
            # Clamp donor to threshold (Damping)
            self.vote_buffer[active_mask] = self.vote_buffer[active_mask].sign() * friction_reference
            
            return gamma_energy.item()

    def absorb_gamma_energy(self, total_pool, friction_reference=128):
        """
        Protocol v5.93: GAMMA INJECTION (The Receiver Kernel)
        Applies shared energy from the global pool to the local qualifiers.
        """
        with torch.no_grad():
            abs_votes = self.vote_buffer.abs()
            
            # RECEIVER MASK: Targeted 'silent' precincts
            # Protocol v5.95: Cold-Start allowed (0.0% to 15.0%)
            low_bound = int(0.00 * friction_reference)
            high_bound = int(0.15 * friction_reference)
            
            # Qualifier mask
            receiver_mask = (abs_votes >= low_bound) & (abs_votes <= high_bound)
            num_receivers = receiver_mask.sum().item()
            
            if num_receivers > 0:
                # 40% Governor limit (always computed first to use as clamp)
                governor_limit = int(0.40 * friction_reference)
                
                # [OVERFLOW PATCH v6.13] Clamp share to governor_limit BEFORE
                # computing boosted_mag — prevents int16 overflow when pool is
                # massive (e.g. 2.88e9 / small num_receivers = millions).
                
                # --- LEGACY PRESERVATION (Pre-v6.13) ---
                # share = int(total_pool / num_receivers)
                # if share > 0:
                #     receivers = self.vote_buffer[receiver_mask]
                #     signs = receivers.sign()
                #     signs[signs == 0] = 1 
                #     boosted_mag = receivers.abs() + share
                #     governor_limit = int(0.40 * friction_reference)
                #     boosted_mag = torch.min(boosted_mag.long(), torch.tensor(governor_limit, device=self.vote_buffer.device)).to(torch.int16)
                #     self.vote_buffer[receiver_mask] = signs * boosted_mag
                # ----------------------------------------

                share = min(int(total_pool / num_receivers), governor_limit)
                if share > 0:
                    receivers = self.vote_buffer[receiver_mask]
                    signs = receivers.sign()
                    signs[signs == 0] = 1

                    boosted_mag = receivers.abs() + share

                    # Governor clamp (share is already <= governor_limit, but
                    # abs() + share could still slightly exceed for saturated receivers)
                    boosted_mag = torch.clamp(boosted_mag.long(), 0, governor_limit).to(torch.int16)

                    # Store back
                    self.vote_buffer[receiver_mask] = signs * boosted_mag
            
            return num_receivers

    def get_consensus_ratio(self):
        """
        PROTOCOL v6.00: THE HONEST SENSOR (NIGHT SHIFT)
        Strictly flattens multi-dimensional QKV attention tensors into a 1D line.
        Matches the backend math to the 1D visual map.
        """
        total_precincts = 0
        locked_precincts = 0
        threshold = getattr(self, 'supermajority_threshold', 20)
        
        # Directive: Iterate through all named parameters in this gate
        for name, param in self.named_parameters():
             # Directive: Locate the Antigravity precinct_votes buffer/attribute
             if hasattr(param, 'precinct_votes'):
                 # CRUCIAL: Forcefully flatten using .view(-1) per v6.00 directive
                 flat_votes = param.precinct_votes.data.abs().view(-1)
                 
                 total_precincts += flat_votes.numel()
                 # Count blocks that pass the supermajority
                 locked_precincts += (flat_votes >= threshold).sum().item()
                 
        # Fallback for modules that store votes in buffers instead of param attributes
        if total_precincts == 0:
            for m in self.modules():
                if hasattr(m, 'vote_buffer'):
                    v = m.vote_buffer.data.abs().view(-1)
                    total_precincts += v.numel()
                    locked_precincts += (v >= threshold).sum().item()

        if total_precincts == 0: 
            return 0.0
            
        return locked_precincts / total_precincts
