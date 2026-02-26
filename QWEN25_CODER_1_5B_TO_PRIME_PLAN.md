# OPERATION: QWEN2.5-CODER-1.5B → PRIME SYSTEM CONVERSION

## Protocol: "The Coder's Gentrification"

### Target: Qwen/Qwen2.5-Coder-1.5B-Instruct

### Author: BatteryPhil / Project Trinity Research

### Date: 2026-02-22

---

## WHY THIS TARGET IS BETTER THAN QWEN3-1.7B

| Advantage | Detail |
|-----------|--------|
| ✅ Smaller hidden dim (1536 vs 2048) | 44% fewer params per projection → 44% smaller vote buffers |
| ✅ Extreme GQA (12Q / 2KV heads) | KV projections are only `(1536, 256)` — almost free |
| ✅ Code-specialized weights | Sparse, structured activations → cleaner LUT snap |
| ✅ Tied embeddings | `lm_head.weight == embed_tokens.weight` — one source, two PRIME buffers |
| ✅ Available as -Instruct | Pre-tuned for chat, instant Answer-Masked SFT baseline |
| ✅ ~3.0GB FP16 on disk | Fits easily in 11GB VRAM alongside activations |

---

## ARCHITECTURE REFERENCE

| Parameter             | Value         | Notes                              |
|-----------------------|---------------|------------------------------------|
| **Model Family**      | Qwen2         | Qwen2.5-Coder uses Qwen2 arch      |
| **Layers**            | 28            |                                    |
| **Hidden Size**       | 1536          |                                    |
| **Num Q Heads**       | 12            |                                    |
| **Num KV Heads**      | 2             | Extreme GQA — 6:1 ratio            |
| **Head Dim**          | 128           | (1536 / 12)                        |
| **KV Proj Width**     | 256           | (128 × 2 KV heads)                 |
| **Intermediate**      | 8960          | SwiGLU MLP                        |
| **Vocab Size**        | 151,936       |                                    |
| **Max Seq Len**       | 131,072       | We'll use 512 for PRIME training   |
| **RoPE Theta**        | 1,000,000     | Kept in FP32, outside vote scope   |
| **Norm Type**         | RMSNorm       | Stays FP32, not quantized          |
| **Activation**        | SwiGLU        | F.silu(gate) * up — not quantized  |
| **Tied Embeddings**   | YES           | Must be untied for PRIME           |
| **Sliding Window**    | NO            | use_sliding_window: false           |
| **Parameters**        | ~1.5B         |                                    |

---

## PROJECTION DIMENSIONS — THE FULL MAP

```
ATTENTION (×28 layers):
  q_proj:    [1536 → 1536]   12 Q-heads × 128 head_dim
  k_proj:    [1536 → 256]     2 KV-heads × 128 head_dim  ← TINY
  v_proj:    [1536 → 256]     2 KV-heads × 128 head_dim  ← TINY
  o_proj:    [1536 → 1536]

MLP (×28 layers):
  gate_proj: [1536 → 8960]   ← Largest per-layer weight
  up_proj:   [1536 → 8960]   ← Largest per-layer weight
  down_proj: [8960 → 1536]

EMBEDDINGS (untied in PRIME):
  embed_tokens → TrueShadowlessEmbedding(151936, 1536)
  lm_head      → GhostLinearTandem(1536, 151936)
```

---

## VRAM BUDGET (GTX 1080 Ti — 11GB)

### Vote Buffer Memory (int16 × 2 bytes)

```
Attention per layer:
  q_proj:    1536 × 1536 × 2B = 4.72 MB
  k_proj:    1536 × 256  × 2B = 0.79 MB
  v_proj:    1536 × 256  × 2B = 0.79 MB
  o_proj:    1536 × 1536 × 2B = 4.72 MB
  Attn total: 10.9 MB × 28 = 305 MB

MLP per layer:
  gate_proj: 1536 × 8960 × 2B = 27.5 MB
  up_proj:   1536 × 8960 × 2B = 27.5 MB
  down_proj: 8960 × 1536 × 2B = 27.5 MB
  MLP total: 82.5 MB × 28    = 2,310 MB

Embeddings (if in VRAM):
  embed_tokens vote_buffer:  151936 × 1536 × 2B = 467 MB
  lm_head vote_buffer:       1536 × 151936 × 2B = 467 MB

TOTAL (no embeddings in VRAM): ~2.6 GB in vote buffers
TOTAL (with embeddings):       ~3.5 GB in vote buffers

Model weights (base_idx + fine_idx at uint8 × 2):
  Transformer layers:  ~1.5B params × 2B = ~3.0 GB
  Scales (FP32):       ~1.5B params × 4B = but only one scale per OUT row
                       (28 × 7 layers × max 8960 outputs) × 4B = ~7 MB (negligible)

TOTAL ESTIMATED VRAM USAGE:
  Model weights:         ~3.0 GB
  Vote buffers:          ~2.6 GB (embedding vote buffers kept on CPU)
  Activations (seq=512): ~1.5 GB
  Optimizer (Adam):      ~0.0 GB (Adam does not train PRIME params — no grad)
  GradScaler + overhead: ~0.5 GB
  ========================
  TOTAL:                 ~7.6 GB   ← FITS on 11GB
```

**Key decision:** Keep `embed_tokens` and `lm_head` vote buffers on CPU. They are only
needed during `apply_votes()` which happens out-of-graph. This frees 934MB VRAM.

---

## THE TIED EMBEDDING PROBLEM

Qwen2.5-Coder uses `tie_word_embeddings: true`:

```python
model.lm_head.weight is model.model.embed_tokens.weight  # True
```

PRIME cannot tie these — they need separate `base_idx / fine_idx / vote_buffer` triplets
because their voting gradients flow in opposite directions and at different scales.

**Solution: Untie during import.**

- Init BOTH from the same Qwen source weight matrix
- Track votes independently from step 1
- The model will naturally re-correlate them as training progresses

---

## PHASE 1: INFRASTRUCTURE

### File 1.1 — `qwen25_prime_wrapper.py`

The core PRIME model that wraps Qwen2's architecture with Ghost layers.

#### A) Weight normalization helper

```python
def normalize_weight_for_lut(weight: torch.Tensor):
    """
    Per-row L-infinity normalization to bring weights into [-1, 1].
    Returns (normalized_weight, scale_per_row).
    The scale becomes each GhostLinearTandem.scale init value.
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
    left = (idx - 1).clamp(0, len(lut)-1)
    d_left  = (flat - lut[left]).abs()
    d_right = (flat - lut[idx]).abs()
    idx = torch.where(d_left < d_right, left, idx).to(torch.int32)
    idx = idx.reshape(weight.shape)
    return torch.div(idx, 256, rounding_mode='floor').to(torch.uint8), (idx % 256).to(torch.uint8)
```

#### B) `PrimeQwen2Attention`

Replaces `Qwen2Attention`. Swaps projections, keeps RoPE and GQA logic.

```
Projections:
  q_proj: GhostLinearTandem(1536, 1536)
  k_proj: GhostLinearTandem(1536, 256)
  v_proj: GhostLinearTandem(1536, 256)
  o_proj: GhostLinearTandem(1536, 1536)

GQA repeat_kv:
  k, v shaped as (B, 2, T, 128) → repeat 6× → (B, 12, T, 128)

RoPE:
  Kept in FP32 via transformers' Qwen2RotaryEmbedding / apply_rotary_pos_emb
  NOT inside any GhostLinearTandem — RoPE is pure positional arithmetic
```

#### C) `PrimeQwen2MLP`

Replaces `Qwen2MLP`.

```
gate_proj: GhostLinearTandem(1536, 8960)
up_proj:   GhostLinearTandem(1536, 8960)
down_proj: GhostLinearTandem(8960, 1536)

Forward:
  gate = self.gate_proj(x, lut=LUT)
  up   = self.up_proj(x, lut=LUT)
  x    = self.down_proj(F.silu(gate) * up, lut=LUT)
```

#### D) `PrimeQwen2DecoderLayer`

Standard residual block:

```
  residual = x
  x = self.input_layernorm(x)           # RMSNorm, FP32
  x = self.self_attn(x, ...)            # PrimeQwen2Attention
  x = residual + x
  residual = x
  x = self.post_attention_layernorm(x)  # RMSNorm, FP32
  x = self.mlp(x)                       # PrimeQwen2MLP
  x = residual + x
```

#### E) `PrimeQwen2CausalLM` — Full Model

```python
self.embed_tokens = TrueShadowlessEmbedding(151936, 1536, lut=LUT)
self.layers       = nn.ModuleList([PrimeQwen2DecoderLayer() for _ in range(28)])
self.norm         = RMSNorm(1536)  # FP32 standard
self.lm_head      = GhostLinearTandem(1536, 151936, lut=LUT, sensitivity=0.05)
self.output_tamer = AdaptiveVarianceTamer(base_threshold=6.0, max_threshold=16.0)
```

Forward pass:

```python
def forward(self, input_ids, targets=None, temperature=1.0):
    x = self.embed_tokens(input_ids)
    for layer in self.layers:
        x = layer(x, ...)
    x = self.norm(x)
    logits = self.lm_head(x, lut=LUT)
    logits = self.output_tamer(logits)
    if temperature != 1.0:
        logits = logits / temperature
    loss = F.cross_entropy(logits.view(-1, 151936), targets.view(-1)) if targets else None
    return logits, loss
```

---

### File 1.2 — `qwen25_prime_importer.py`

Runs ONCE. Reads Qwen FP32 weights, snaps to LUT, writes PRIME checkpoint.

**Memory strategy:** Run entirely on CPU. Peak RAM ~6GB.

**Layer-by-layer import loop:**

```
For each of 28 layers:
  Load qwen.model.layers[i] (stays on CPU)
  For each of 7 projections (q,k,v,o, gate, up, down):
    1. Extract weight[out_features, in_features]
    2. per-row normalize → (norm_w, scale)
    3. fp16_weight_to_lut_index(norm_w, lut) → (base_idx, fine_idx)
    4. Copy into prime.layers[i].<proj>.base_idx / fine_idx / scale
  Copy RMSNorm weights directly (no quantization)
  Log snapping coverage diagnostics

For embed_tokens:
  Initialize TrueShadowlessEmbedding from qwen.model.embed_tokens.weight

For lm_head (untied):
  Initialize GhostLinearTandem(1536, 151936) from same source weight (transposed)

For final norm:
  Copy directly
```

**Diagnostics to log:**

- % of weights that needed clamping (were outside [-1, 1] before normalization)
- Distribution of snapped LUT indices (expect bell-curve centred around 32768)
- Any layer that is pathologically uniform (could indicate dead weights)

**Save:** `qwen25_coder_prime_init.pt`
**Expected size:** ~4.5 GB (uint8 × 2 per weight + FP32 scales + RMSNorm)

---

### File 1.3 — `verify_qwen25_prime.py`

Quick sanity check after import. Checks:

1. All `base_idx` buffers are non-trivial (index distribution is not uniform-at-zero)
2. Forward pass on dummy input produces finite logits
3. Logit range is reasonable (e.g., [-10, 10])
4. All `vote_buffer` tensors are zeroed (no phantom votes)
5. Loss on a dummy batch of known tokens is finite

```python
model = PrimeQwen2CausalLM(lut=lut)
model.load_state_dict(torch.load("qwen25_coder_prime_init.pt"))
model.eval()

dummy_ids     = torch.randint(0, 151936, (1, 64))
dummy_targets = dummy_ids.roll(-1)
with torch.no_grad():
    logits, loss = model(dummy_ids, targets=dummy_targets)

assert not torch.isnan(logits).any()
assert not torch.isinf(logits).any()
assert loss.item() < 50.0,  f"Loss {loss.item():.2f} — suspect import failure"
print(f"[VERIFY] Loss={loss.item():.4f} | Logit range [{logits.min():.2f}, {logits.max():.2f}]")
```

If the imported model is working, the initial loss should be near `ln(151936) ≈ 11.93`
(uniform distribution) — possibly LOWER if the Qwen weights survived snapping intact,
meaning the model immediately "recognizes" coherent text without any training.

---

## PHASE 2: TRAINING LOOP

### File 2.1 — `qwen25_prime_train.py` (Fork of `primal_train_modular.py`)

#### CONFIG

```python
CONFIG = {
    'vocab_size':   151936,
    'seq_len':      512,        # Conservative start. Can increase to 1024 once stable.
    'dim':          1536,
    'n_layers':     28,
    'n_heads':      12,
    'n_kv_heads':   2,
    'intermediate': 8960,
    'lr':           1e-4,
    'batch_size':   1,
    'grad_accum':   256,
    'mode':         'primal',
    'device':       'cuda' if torch.cuda.is_available() else 'cpu',
    'stats_file':   'stats_qwen25_coder.json',
    'prefix':       'qwen25_coder',
    'checkpoint':   'qwen25_coder_prime_init.pt',
}
```

#### Tokenizer

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
# pad_token: tokenizer.eos_token (151645)
# Use chat template for instruction data:
# <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
```

#### Answer-Masked Loss (replicating PAPER.md methodology)

For instruction fine-tuning, only compute loss on the assistant response tokens:

```python
def answer_masked_loss(logits, targets, response_mask):
    """
    Only backprop through the assistant response portion.
    Masks the user prompt tokens to prevent easy syntax learning.
    """
    logits_flat  = logits.view(-1, 151936)
    targets_flat = targets.view(-1)
    mask_flat    = response_mask.view(-1).bool()

    logits_masked  = logits_flat[mask_flat]
    targets_masked = targets_flat[mask_flat]

    if logits_masked.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return F.cross_entropy(logits_masked, targets_masked)
```

#### Dataset Strategy

**Phase 1 (steps 0–500): Stabilization**

- Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- Goal: Confirm vote buffers accumulate, loss descends from ~11.93
- Loss target: < 6.0

**Phase 2 (steps 500–1500): Code Instruction Fine-tuning**

- Dataset: [nampdn-ai/tiny-codes](https://huggingface.co/datasets/nampdn-ai/tiny-codes)
  or [codeparrot/github-code](https://huggingface.co/datasets/codeparrot/github-code)
- Use Answer-Masked loss (mask on `<|im_start|>assistant` sections)
- Goal: Reconnect coder knowledge graph broken by quantization
- Loss target: < 4.0

**Phase 3 (steps 1500+): Perplexity refinement**

- Dataset: Mixed TinyStories + code
- Reduce LR as Night Shift Consensus rises
- Loss target: < 3.5

#### Reused Components (ZERO CHANGES NEEDED)

```
AntigravitySentinelV571     → Works as-is (iterates all GhostLinearTandem)
NightShiftSupervisor        → Works as-is (Master Lock + Layer Peel below)
GammaController             → Works as-is
ThermalVigil                → Works as-is
map_voter_activity()        → Works as-is
break_manifold_stall()      → Works as-is
GradScaler                  → Works as-is (AMP for forward, integer for votes)
```

#### Gradient Checkpointing

Must be enabled — 28 layers × (1536² + 1536×8960) is significant:

```python
# In PrimeQwen2DecoderLayer.forward():
if self.gradient_checkpointing and self.training:
    return checkpoint(self._forward_impl, hidden_states, ..., use_reentrant=False)
```

---

## PHASE 3: PROTOCOL v7.00 — LAYER PEEL

### The Problem

With 28 layers all voting simultaneously from step 1, we risk a synchronized multi-layer
avalanche when the Master Lock releases at step 1000. This is more dangerous than with
`GhostGPT` (which had 2 Ghost layers total vs our 28 × 7 = 196 Ghost layers here).

### The Solution: Top-Down Layer Unlock

```
Steps     0 –  200:  ALL 28 layers LOCKED (vote buffers accumulate, zero flips)
                     Purpose: let pressure build uniformly from the signal.

Steps   200 –  250:  Layer 27 (topmost — closest to lm_head) unlocked
Steps   250 –  300:  Layer 26 unlocked
Steps   300 –  350:  Layer 25 unlocked
...etc — one layer per 50 steps...
Steps  1,550+:       All 28 layers + lm_head + embed_tokens fully active.
                     Normal Night Shift consensus tracking resumes.
```

**Why top-down?**
The output head sees the clearest gradient signal (direct loss back-prop, single softmax).
Layers closer to the bottom (embedding layers) get the noisiest, most attenuated gradient
signal. By crystallizing top-down, each lower layer gets a progressively cleaner signal
from the already-crystallized layer above it.

### Implementation in `NightShiftSupervisor`

```python
def get_unlocked_layers(self, current_step):
    """Returns set of layer indices that are allowed to flip."""
    if current_step < 200:
        return set()  # Full lock

    unlocked = set()
    for i in range(28):
        layer_from_top = 27 - i  # 27, 26, 25... 0
        unlock_at_step = 200 + (i * 50)
        if current_step >= unlock_at_step:
            unlocked.add(layer_from_top)
    return unlocked

def get_lm_head_unlocked(self, current_step):
    """LM head unlocks with Layer 27 at step 200."""
    return current_step >= 200

def get_embeddings_unlocked(self, current_step):
    """Embeddings unlocked last — step 1600 (after all transformer layers)."""
    return current_step >= 1600
```

Modify `AntigravitySentinelV571.apply_safestep_and_vote()`:

```python
unlocked_layers = self.supervisor.get_unlocked_layers(current_step)
for name, module in self.model.named_modules():
    if isinstance(module, GhostLinearTandem):
        # Parse layer index from name (e.g., "layers.15.mlp.gate_proj")
        layer_idx = parse_layer_idx(name)  # returns int or None
        if layer_idx is not None and layer_idx not in unlocked_layers:
            stats = {"flips": 0, "avg_stride": 0, "max_stride": 0}
        elif name == "lm_head" and not self.supervisor.get_lm_head_unlocked(current_step):
            stats = {"flips": 0, "avg_stride": 0, "max_stride": 0}
        else:
            stats = module.apply_tandem_votes(learning_rate=current_lr)
```

---

## PHASE 4: VALIDATION & COMPRESSION

### 4.1 — Benchmarks

Use HumanEval and MBPP (standard code benchmarks) since this is a coder model:

- **HumanEval pass@1** (baseline Qwen2.5-Coder-1.5B-Instruct: ~37%)
- **Target after PRIME:** Recover ≥ 25% (Phase 1), ≥ 32% (Phase 2)
- **Perplexity on Python code** (GitHub-Code held-out): target < 15

### 4.2 — Inference with Qwen Chat Template

```python
# Using Qwen2.5-Coder-1.5B-Instruct format
messages = [{"role": "user", "content": "Write a binary search in Python."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# <|im_start|>user\nWrite a binary search in Python.<|im_end|>\n<|im_start|>assistant\n
```

### 4.3 — PRIME Packing Estimate

```
1.5B params × (base_idx: 1B + fine_idx: 1B) = 2.0B bytes = ~1.9 GB raw
After delta-encoding adjacent LUT indices: ~500-700 MB packed
vs original FP16: 3.0 GB
Compression ratio: ~4-6×
```

---

## IMPLEMENTATION CHECKLIST (Ordered)

```
[ ] PHASE 1: INFRASTRUCTURE
    [ ] 1.1  qwen25_prime_wrapper.py
              [ ] normalize_weight_for_lut()
              [ ] fp16_weight_to_lut_index()
              [ ] PrimeQwen2Attention (q, k, v, o with GQA + RoPE)
              [ ] PrimeQwen2MLP (gate, up, down with SwiGLU)
              [ ] PrimeQwen2DecoderLayer (with gradient checkpointing flag)
              [ ] PrimeQwen2CausalLM (untied embed + lm_head, output_tamer)
    [ ] 1.2  qwen25_prime_importer.py
              [ ] Load Qwen2.5-Coder-1.5B-Instruct (FP32 on CPU)
              [ ] Per-row normalization + LUT snapping (all 28 × 7 projections)
              [ ] RMSNorm weight copy
              [ ] Untied embedding import (same source, separate buffers)
              [ ] Coverage diagnostics (% out-of-range, index distribution)
              [ ] Save: qwen25_coder_prime_init.pt
    [ ] 1.3  verify_qwen25_prime.py
              [ ] Forward pass finite check
              [ ] Initial loss check (expect near 11.93)
              [ ] Vote buffer zero check
              [ ] base_idx distribution histogram

[ ] PHASE 2: TRAINING LOOP
    [ ] 2.1  qwen25_prime_train.py (fork primal_train_modular.py)
              [ ] CONFIG with correct dims
              [ ] Qwen2.5 tokenizer
              [ ] Answer-masked loss function
              [ ] Gradient checkpointing per layer
              [ ] TinyStories Phase 1 loader
              [ ] VRAM smoke test (1 forward pass at seq=512)
              [ ] 10-step run — confirm vote buffers accumulating non-zero values
              [ ] 100-step run — confirm at least some flips after unlock at step 200

[ ] PHASE 3: PROTOCOL v7.00 — LAYER PEEL
    [ ] 3.1  NightShiftSupervisor.get_unlocked_layers(step)
    [ ] 3.2  AntigravitySentinel layer-lock enforcement
    [ ] 3.3  Embedding unlock at step 1600
    [ ] 3.4  500-step training run with Layer Peel — validate no avalanche

[ ] PHASE 4: VALIDATION
    [ ] 4.1  HumanEval pass@1 benchmark
    [ ] 4.2  Perplexity on TinyStories + Python code
    [ ] 4.3  Interactive chat test (Qwen chat template)
    [ ] 4.4  PRIME packing (estimate compressed size)
    [ ] 4.5  Update PAPER.md with Qwen2.5-Coder-1.5B results
```

---

## RISK REGISTER

| Risk | Severity | Mitigation |
|------|----------|------------|
| VRAM OOM during import | HIGH | Run on CPU (all 6GB fits in 16GB RAM) |
| MLP vote buffers (2.3GB) too large | HIGH | Keep embed vote buffers on CPU; train one layer group at a time if needed |
| SwiGLU gate couples gate+up gradients | MEDIUM | Both get separate vote buffers (independent) — coupling only in forward pass, not backward |
| Tied-to-untied embedding divergence | MEDIUM | Normal — PRIME trains them apart. Not a bug. |
| RoPE theta=1M breaks with seq=512 | LOW | RoPE with any theta works fine at shorter sequences |
| GQA repeat_kv adds a non-differentiable op | LOW | repeat_kv is just `.expand()` — gradients flow through correctly |
| Initial loss >> 11.93 | MEDIUM | Indicates scale initialization is off — recheck normalize_weight_for_lut() |

---

## QUICK-START COMMANDS (After Files Are Written)

```bash
# Step 1: Import Qwen2.5-Coder weights into PRIME format
python qwen25_prime_importer.py --device cpu --output qwen25_coder_prime_init.pt

# Step 2: Verify the checkpoint is healthy
python verify_qwen25_prime.py

# Step 3: Begin Phase 1 training
python qwen25_prime_train.py --manifold unified --resume

# Step 4: Monitor (existing tools work unchanged)
python monitor_primal.py

# Step 5: Benchmark after 500 steps
python primal_bench.py --checkpoint qwen25_coder_prime_500.pt
```

---

## WHY THIS WILL WORK — THE PHYSICS

**The core bet:** Qwen2.5-Coder's weights, after 1.5B parameters of code training,
live in a distribution where most weights are small (< 0.1 in magnitude). The
Prime-Harmonic LUT has its highest density of points near zero. So the vast majority
of Qwen's weights will snap to LUT coordinates with < 2% relative quantization error.

The outliers (large weights, mostly in MLP gate/up projections) get normalized per-row
before snapping — their magnitude is stored in the `scale` parameter, recovering
exact representation on dequantization.

The voting engine then refines every weight independently from its initial snapped
coordinate, guided by actual code generation loss gradients. Qwen's pre-existing
knowledge isn't lost — it's the **starting grid position** for every voter precinct.

The hard problem is the 28-layer avalanche risk. Protocol v7.00 Layer Peel solves this
by converting a potentially catastrophic simultaneous event into a controlled waterfall
of crystallization, one layer per 50 steps, top to bottom.

---
*End of Plan. Begin with Phase 1.1: `qwen25_prime_wrapper.py`.*
