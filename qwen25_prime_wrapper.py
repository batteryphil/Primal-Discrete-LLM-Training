import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from ghost_core import GhostLinearTandem, AdaptiveVarianceTamer

# ==============================================================================
# QWEN2.5-CODER-1.5B PRIME WRAPPER
# ==============================================================================
# Architecturally aligned with Qwen2.5-Coder-1.5B-Instruct.
# Replaces nn.Linear with GhostLinearTandem for Prime-Evolved Quantization.
# ==============================================================================

def normalize_weight_for_lut(weight: torch.Tensor):
    """
    Per-row L-infinity normalization to bring weights into [-1, 1].
    Returns (normalized_weight, scale_per_row).
    """
    scale = weight.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
    return weight / scale, scale

def fp16_weight_to_lut_index(weight: torch.Tensor, lut: torch.Tensor):
    """
    Snaps a weight matrix to nearest LUT index.
    Returns base_idx (uint8), fine_idx (uint8).
    Weight must already be in [-1, 1] range.
    """
    flat = weight.reshape(-1).float()
    idx = torch.searchsorted(lut.contiguous(), flat.contiguous()).clamp(0, len(lut)-1)
    
    # Check left neighbor for true nearest point
    left = (idx - 1).clamp(0, len(lut)-1)
    d_left  = (flat - lut[left]).abs()
    d_right = (flat - lut[idx]).abs()
    idx = torch.where(d_left < d_right, left, idx).to(torch.int32)
    
    idx = idx.reshape(weight.shape)
    return torch.div(idx, 256, rounding_mode='floor').to(torch.uint8), (idx % 256).to(torch.uint8)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=1000000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # cos, sin are [seq_len, dim]
    # position_ids is [batch_size, seq_len]
    cos = cos[position_ids].unsqueeze(1) # [batch, 1, seq, dim]
    sin = sin[position_ids].unsqueeze(1) # [batch, 1, seq, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class PrimeQwen2Attention(nn.Module):
    def __init__(self, config, layer_idx=None, lut=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config['num_key_value_heads']
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = GhostLinearTandem(self.hidden_size, self.num_heads * self.head_dim, lut=lut)
        self.k_proj = GhostLinearTandem(self.hidden_size, self.num_key_value_heads * self.head_dim, lut=lut)
        self.v_proj = GhostLinearTandem(self.hidden_size, self.num_key_value_heads * self.head_dim, lut=lut)
        self.o_proj = GhostLinearTandem(self.num_heads * self.head_dim, self.hidden_size, lut=lut)
        
        self.rotary_emb = Qwen2RotaryEmbedding(self.head_dim, max_position_embeddings=config.get('max_position_embeddings', 2048), base=config.get('rope_theta', 1000000))

    def forward(self, hidden_states, attention_mask=None, position_ids=None, lut=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, lut=lut)
        key_states = self.k_proj(hidden_states, lut=lut)
        value_states = self.v_proj(hidden_states, lut=lut)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # GQA Repeat KV
        key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_key_value_groups)
        value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_key_value_groups)

        # PyTorch Scaled Dot Product Attention (FlashAttention compatible)
        # Handles the sqrt(d) scaling automatically.
        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attn_mask=attention_mask,
            dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output, lut=lut)

        return attn_output

class PrimeQwen2MLP(nn.Module):
    def __init__(self, config, lut=None):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        self.gate_proj = GhostLinearTandem(self.hidden_size, self.intermediate_size, lut=lut)
        self.up_proj = GhostLinearTandem(self.hidden_size, self.intermediate_size, lut=lut)
        self.down_proj = GhostLinearTandem(self.intermediate_size, self.hidden_size, lut=lut)

    def forward(self, x, lut=None):
        gate = self.gate_proj(x, lut=lut)
        up = self.up_proj(x, lut=lut)
        return self.down_proj(F.silu(gate) * up, lut=lut)

class PrimeQwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=None, lut=None):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.self_attn = PrimeQwen2Attention(config, layer_idx=layer_idx, lut=lut)
        self.mlp = PrimeQwen2MLP(config, lut=lut)
        self.input_layernorm = RMSNorm(config['hidden_size'], eps=config.get('rms_norm_eps', 1e-6))
        self.post_attention_layernorm = RMSNorm(config['hidden_size'], eps=config.get('rms_norm_eps', 1e-6))
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, position_ids=None, lut=None):
        if self.gradient_checkpointing and self.training:
            return checkpoint(
                self._forward_impl, 
                hidden_states, 
                attention_mask, 
                position_ids, 
                lut, 
                use_reentrant=False
            )
        return self._forward_impl(hidden_states, attention_mask, position_ids, lut)

    def _forward_impl(self, hidden_states, attention_mask=None, position_ids=None, lut=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, lut=lut)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, lut=lut)
        hidden_states = residual + hidden_states
        return hidden_states

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

class PrimeQwen2CausalLM(nn.Module):
    def __init__(self, config, lut=None):
        super().__init__()
        self.config = config
        self.lut = lut
        self.embed_tokens = TrueShadowlessEmbedding(config['vocab_size'], config['hidden_size'], lut=lut)
        self.layers = nn.ModuleList([PrimeQwen2DecoderLayer(config, layer_idx=i, lut=lut) for i in range(config['num_hidden_layers'])])
        self.norm = RMSNorm(config['hidden_size'], eps=config.get('rms_norm_eps', 1e-6))
        self.lm_head = GhostLinearTandem(config['hidden_size'], config['vocab_size'], lut=lut)
        self.output_tamer = AdaptiveVarianceTamer(base_threshold=6.0, max_threshold=16.0)

    def forward(self, input_ids, labels=None, attention_mask=None, position_ids=None, temperature=1.0):
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, lut=self.lut)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states, lut=self.lut)
        logits = self.output_tamer(logits)
        
        if temperature != 1.0:
            logits = logits / temperature
            
        loss = None
        if labels is not None:
             loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), labels.view(-1))
             
        return logits, loss
