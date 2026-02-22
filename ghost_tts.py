import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- CONFIGURATION (Matches Trinity Training) ---
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cooldown_steps": 5,  # Steps to lock a weight after flipping
}

# --- 1.58-BIT QUANTIZATION ENGINE (GHOST) ---

import manifolds

# Global LUT for quantization (will be generated on init)
PRIMAL_LUT = manifolds.generate_linear_manifold(device='cuda' if torch.cuda.is_available() else 'cpu')

class GhostQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, scale, lut, vote_buffer, sensitivity):
        # 1. Dequantize: Look up values in LUT and scale
        weights = lut[indices.long()] * scale
        
        ctx.save_for_backward(indices, scale, lut, vote_buffer)
        ctx.sensitivity = sensitivity 
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        indices, scale, lut, vote_buffer = ctx.saved_tensors
        sensitivity = ctx.sensitivity
        
        # [NEW] Gradient Centering (Antigravity)
        # Prevents global shifts from dominating the "vote"
        grad_output = grad_output - grad_output.mean(dim=1, keepdim=True)
        
        # 1. Gradient for Scale (Summed over input dim for per-channel)
        weights_proxy = lut[indices.long()]
        grad_scale = (grad_output * weights_proxy).sum(dim=1, keepdim=True)
        
        # 2. THE WHISPER LOGIC (Adaptive Thresholding)
        # We want to find which weights *should* flip based on gradient direction
        direction = -torch.sign(grad_output * scale.sign()).to(torch.int8)
        
        # Adaptive Threshold
        grad_abs = torch.abs(grad_output)
        g_mean = grad_abs.mean()
        g_std = grad_abs.std()
        
        # Sensitivity controls how "quiet" a gradient can be and still vote
        threshold = g_mean + (sensitivity * g_std)
        
        # The Vote
        significant_mask = (grad_abs > threshold).to(torch.int8)
        votes = direction * significant_mask
        vote_buffer.add_(votes)
        
        # We return gradients for inputs (indices, scale, lut, vote_buffer, sensitivity)
        # Only scale needs a gradient here. Indices are discrete.
        return None, grad_scale, None, None, None

class GhostLinear(nn.Module):
    def __init__(self, in_features, out_features, sensitivity=0.15):
        super().__init__()
        self.sensitivity = sensitivity 
        
        # Initialize with small random weights
        raw_w = torch.randn(out_features, in_features) * 0.02
        
        # Quantize immediately to indices
        self.register_buffer('grid_indices', self.quantize_to_indices(raw_w))
        
        # Audit Buffers (Vote Buffer & Cooldown)
        self.register_buffer('vote_buffer', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('cooldown', torch.zeros(out_features, in_features, dtype=torch.uint8))
        self.register_buffer('steps', torch.zeros(1, dtype=torch.long))
        
        # Scale Parameter (Trainable)
        self.scale = nn.Parameter(torch.ones(out_features, 1))

    def quantize_to_indices(self, weights):
        chunk_size = 1024
        num_chunks = math.ceil(weights.shape[0] / chunk_size)
        indices_list = []
        w_gpu_full = weights.to(CONFIG['device'])
        
        # Use simple distance minimization to find closest LUT index
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, weights.shape[0])
            w_chunk = w_gpu_full[start:end]
            
            # Distance matrix: |w - lut|
            # w_chunk: [Chunk, In], LUT: [256]
            diff = torch.abs(w_chunk.unsqueeze(-1) - PRIMAL_LUT)
            chunk_indices = torch.argmin(diff, dim=-1).to(torch.uint8)
            indices_list.append(chunk_indices)
            
        return torch.cat(indices_list, dim=0).cpu()

    def forward(self, x, annealing_factor=1.0):
        effective_sensitivity = self.sensitivity * annealing_factor
        
        # Ensure LUT is on correct device
        global PRIMAL_LUT
        if PRIMAL_LUT.device != x.device:
            PRIMAL_LUT = PRIMAL_LUT.to(x.device)
            
        weights = GhostQuantFunction.apply(
            self.grid_indices, self.scale, PRIMAL_LUT, self.vote_buffer, effective_sensitivity
        )
        return F.linear(x, weights)

    def apply_votes(self, adaptive_prob, name="Unknown"):
        # This function is called by the optimizer/training loop to apply flipped bits
        # 1. Hysteresis (Cooldown)
        with torch.no_grad():
            mask = self.cooldown > 0
            self.cooldown[mask] -= 1
            
        # 2. Gravity Well (Pull scales to 1.0)
        with torch.no_grad():
            self.scale.data = self.scale.data * 0.999 + (1.0 * 0.001)

        if self.vote_buffer.abs().max() == 0:
            return 0
            
        # 3. Resonator & Floodgate
        prob_tensor = torch.full_like(self.vote_buffer.float(), adaptive_prob)
        high_pressure_mask = self.vote_buffer.abs() >= 32
        prob_tensor[high_pressure_mask] = 0.1 
        
        # [SENTINEL] Conviction Override
        saturated_mask = self.vote_buffer.abs() >= 16
        prob_tensor[saturated_mask] = 1.0
        
        # 4. Stochastic Flip
        final_direction = torch.sign(self.vote_buffer).to(torch.int8)
        flip_mask = (torch.rand_like(self.vote_buffer.float()) < prob_tensor).to(torch.int8)
        ready_mask = (self.cooldown == 0).to(torch.int8)
        valid_flips = final_direction * flip_mask * ready_mask
        
        # 5. Apply Updates
        new_indices = self.grid_indices.int() + valid_flips.int()
        self.grid_indices.copy_(new_indices.clamp(0, 255).to(torch.uint8))
        
        # 6. Conviction Reset
        with torch.no_grad():
            self.vote_buffer[valid_flips != 0] = 0 
            
        # 7. Dynamic Cooldown
        num_flips = torch.count_nonzero(valid_flips).item()
        if num_flips > 0:
            if num_flips > (self.grid_indices.numel() * 0.001):
                lock_duration = CONFIG['cooldown_steps'] * 3
            else:
                lock_duration = CONFIG['cooldown_steps']
            self.cooldown[valid_flips != 0] = lock_duration
        
        # 8. Linear Friction (Reduced: Every 10 steps)
        self.steps += 1
        if self.steps.item() % 10 == 0:
            with torch.no_grad():
                pos_mask = self.vote_buffer > 0
                neg_mask = self.vote_buffer < 0
                self.vote_buffer[pos_mask] -= 1
                self.vote_buffer[neg_mask] += 1
        
        return num_flips

# --- VARIANCE ADAPTOR (Duration, Pitch, Energy) ---

class VariancePredictor(nn.Module):
    def __init__(self, model_dim, filter_size=256, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(model_dim, filter_size, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.LayerNorm(filter_size)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.LayerNorm(filter_size)
        self.dropout2 = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(filter_size, 1)

    def re_init(self):
        """Specifically used by The Sentinel Protocol to wake up flatlined predictors."""
        for m in [self.conv1, self.conv2, self.linear_layer]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        # x: [Batch, Time, Dim]
        
        # Block 1
        x = x.transpose(1, 2) # [B, C, T]
        x = self.conv1(x)
        x = F.relu(x)
        x = x.transpose(1, 2) # [B, T, C]
        x = self.norm1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = x.transpose(1, 2) # [B, C, T]
        x = self.conv2(x)
        x = F.relu(x)
        x = x.transpose(1, 2) # [B, T, C]
        x = self.norm2(x)
        x = self.dropout2(x)
        
        # Output
        x = self.linear_layer(x)
        
        if mask is not None:
             x = x.masked_fill(mask.unsqueeze(-1), 0.0)
             
        return x.squeeze(-1)

class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, duration, max_len=None):
        output = []
        # Check inputs
        batch_size = x.shape[0]
        
        if duration is None:
             return x # Should logic handle this?
             
        for i in range(batch_size):
            repeats = duration[i].long()
            # Simple expansion
            expanded = torch.repeat_interleave(x[i], repeats, dim=0)
            output.append(expanded)
            
        # Pad to max_len
        if max_len is None:
            max_len = max([t.shape[0] for t in output]) if output else 0
            
        padded_output = torch.zeros(batch_size, max_len, x.shape[2]).to(x.device)
        for i, t in enumerate(output):
            length = min(t.shape[0], max_len)
            padded_output[i, :length] = t[:length]
            
        return padded_output

# --- GHOST TRANSFORMER BLOCK ---

class GhostTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        
        # Attention
        self.q_proj = GhostLinear(dim, dim)
        self.k_proj = GhostLinear(dim, dim)
        self.v_proj = GhostLinear(dim, dim)
        self.out_proj = GhostLinear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn_up = GhostLinear(dim, dim * 4)
        self.ffn_down = GhostLinear(dim * 4, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [Batch, Time, Dim]
        B, T, C = x.shape
        
        # Attention
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
             # mask: [Batch, Time] -> [Batch, 1, 1, Time]
             # Assuming mask is 1 for padding (True)
             pass 
             # Actually mask logic needs care.
             
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        
        out = self.out_proj(out)
        x = residual + self.dropout(out)
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn_up(x)
        x = F.gelu(x)
        x = self.ffn_down(x)
        x = residual + self.dropout(x)
        
        return x

# --- MAIN MODEL (GHOST-TTS) ---

class GhostTTS(nn.Module):
    def __init__(self, vocab_size, model_dim=256, heads=2, layers=4): # Mini config
        super().__init__()
        
        self.model_dim = model_dim
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(1024, model_dim) # Simple positional
        self.encoder_layers = nn.ModuleList([
            GhostTransformerBlock(model_dim, heads) for _ in range(layers)
        ])
        
        # Variance Adaptor
        self.duration_predictor = VariancePredictor(model_dim, filter_size=256)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_dim, filter_size=256)
        self.energy_predictor = VariancePredictor(model_dim, filter_size=256)
        
        self.pitch_projection = nn.Linear(1, model_dim)
        self.energy_projection = nn.Linear(1, model_dim)
        
        # Decoder
        self.decoder_pos_emb = nn.Embedding(2048, model_dim) # Expanded length
        self.decoder_layers = nn.ModuleList([
            GhostTransformerBlock(model_dim, heads) for _ in range(layers)
        ])
        
        # Output
        # Using standard Linear for high-fidelity regression
        self.mel_dense = nn.Linear(model_dim, 80) 
        
        # PostNet (Refines Mel)
        self.postnet = nn.Sequential(
            nn.Conv1d(80, 512, 5, padding=2), nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0.5),
            nn.Conv1d(512, 512, 5, padding=2), nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0.5),
            nn.Conv1d(512, 512, 5, padding=2), nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0.5),
            nn.Conv1d(512, 512, 5, padding=2), nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0.5),
            nn.Conv1d(512, 80, 5, padding=2)
        )

    def forward(self, text, mel_target=None, duration_target=None, pitch_target=None, energy_target=None):
        # text: [B, T_src]
        
        # 1. Encoder
        x = self.embedding(text)
        b, t, c = x.shape
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        x = x + self.pos_emb(positions)
        
        # Masking?
        src_mask = None # Add if padding exists
        
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
            
        encoder_output = x
        
        # 2. Variance Adaptor
        # Duration
        log_duration_prediction = self.duration_predictor(encoder_output, src_mask)
        
        if duration_target is not None:
             duration_rounded = duration_target
        else:
              # Inference: use prediction
             duration_rounded = torch.clamp((torch.exp(log_duration_prediction) - 1), min=2).long()
             
        # Length Regulation
        x = self.length_regulator(encoder_output, duration_rounded)
        
        # Pitch (Continuous Projection)
        pitch_prediction = self.pitch_predictor(x)
        if pitch_target is not None:
             p_in = pitch_target.unsqueeze(-1) # [B, T, 1]
        else:
             p_in = pitch_prediction.unsqueeze(-1) # [B, T, 1]
        p_embed = self.pitch_projection(p_in)
             
        # Energy (Continuous Projection)
        energy_prediction = self.energy_predictor(x)
        if energy_target is not None:
             e_in = energy_target.unsqueeze(-1) # [B, T, 1]
        else:
             e_in = energy_prediction.unsqueeze(-1) # [B, T, 1]
        e_embed = self.energy_projection(e_in)
             
        x = x + p_embed + e_embed
        
        # Decoder
        b_dec, t_dec, c_dec = x.shape
        # Re-create positions for decoder
        dec_positions = torch.arange(t_dec, device=x.device).unsqueeze(0).expand(b_dec, t_dec)
        
        # Ensure pos_emb can handle this length
        if t_dec > self.decoder_pos_emb.num_embeddings:
             # Truncate?
             x = x[:, :self.decoder_pos_emb.num_embeddings, :]
             dec_positions = dec_positions[:, :self.decoder_pos_emb.num_embeddings]
             
        x = x + self.decoder_pos_emb(dec_positions)
        
        for layer in self.decoder_layers:
            x = layer(x) # Todo: Mask?
            
        # Output
        mel_output = self.mel_dense(x)
        
        # PostNet works on [B, C, T]
        mel_output_transposed = mel_output.transpose(1, 2)
        mel_postnet_output = self.postnet(mel_output_transposed).transpose(1, 2)
        mel_postnet_output = mel_output + mel_postnet_output
        
        return (mel_output, mel_postnet_output, log_duration_prediction, pitch_prediction, energy_prediction, src_mask)

    def apply_ghost_votes(self, adaptive_prob=0.01):
        # Propagate vote application to all GhostLinear layers
        total_flips = 0
        for name, module in self.named_modules():
            if isinstance(module, GhostLinear):
                total_flips += module.apply_votes(adaptive_prob, name=name)
        return total_flips
