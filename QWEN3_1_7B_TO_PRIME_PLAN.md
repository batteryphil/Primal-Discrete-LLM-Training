# OPERATION: QWEN3-1.7B → PRIME SYSTEM CONVERSION

## Protocol: "Gentrification at Scale"

### Author: BatteryPhil / Project Trinity Research

### Date: 2026-02-22

---

## EXECUTIVE SUMMARY

We are **evolving** Qwen3-1.7B into the PRIME discrete manifold system — not retraining
it. The core thesis: Qwen's dense FP16 weights already contain latent structure that can
be "snapped" to the 16-bit Prime-Harmonic Manifold using Gradient-Evolved Quantization
via the existing `GhostLinearTandem` voting engine.

This is the **TinyLlama → PRIME** pipeline, but applied to a significantly bigger and
more architecturally complex target (Qwen3 uses GQA, RoPE, SiLU, RMSNorm — none of which
exist in the current `GhostGPT`). Full conversion requires 4 phases.

---

## QWEN3-1.7B ARCHITECTURE REFERENCE

| Parameter           | Value          |
|---------------------|----------------|
| **Layers**          | 28             |
| **Hidden Size**     | 2048           |
| **Num Q Heads**     | 16             |
| **Num KV Heads**    | 8 (GQA)        |
| **Head Dim**        | 128            |
| **Intermediate**    | 6144           |
| **Vocab Size**      | 151,936        |
| **Seq Length**      | 32,768         |
| **RoPE Theta**      | 1,000,000      |
| **Norm Type**       | RMSNorm        |
| **Activation**      | SiLU (in MLP)  |
| **Parameters**      | ~1.7B          |

---

## THE CONVERSION STRATEGY

### What We Are Doing

Instead of the current `GhostGPT` (a custom architecture trained from scratch), we are
building a **PRIME Wrapper** that surgically replaces Qwen3's `nn.Linear` layers with
`GhostLinearTandem` layers — while IMPORTING and PRESERVING Qwen's pre-trained FP16
weights as initialization for the vote buffer starting positions.

This is the "Shadow Weight" technique from the PAPER.md, applied at scale:

- **Forward:** weights snapped to nearest LUT coordinate
- **Backward:** gradients drive votes → `apply_tandem_votes()` updates LUT indices
- **Qwen weights become the initial grid positions**, not random Xavier init

### What We Are NOT Doing

- NOT training from scratch (would nuke Qwen's knowledge)  
- NOT changing the Qwen transformer structure (attention, MLP, RoPE stay intact)  
- NOT using the `PrimalTRMCore` recursive refinement engine (that was custom arch)  
- NOT using GPT2 tokenizer (Qwen3 has vocab_size=151,936)

---

## PHASE 1: INFRASTRUCTURE (NEW FILES NEEDED)

### 1.1 — `qwen3_prime_wrapper.py` (NEW — Core Conversion Module)

This is the most important file. It must:

#### A) `fp16_weight_to_lut_index(weight_tensor, lut)` — Snap FP16 → LUT

```python
def fp16_weight_to_lut_index(weight_tensor, lut):
    """
    Maps each FP16 weight to its nearest LUT coordinate.
    Returns (base_idx [uint8], fine_idx [uint8]) split from 16-bit combined index.
    Uses torch.searchsorted for O(log N) binary search on the sorted LUT.
    """
    weight_flat = weight_tensor.view(-1).float().clamp(-1.0, 1.0)
    combined = torch.searchsorted(lut, weight_flat)
    combined = combined.clamp(0, len(lut) - 1)
    # Check neighbors (same as TrueShadowlessEmbedding does)
    idx_left = (combined - 1).clamp(0, len(lut) - 1)
    dist_left = (weight_flat - lut[idx_left]).abs()
    dist_right = (weight_flat - lut[combined]).abs()
    combined = torch.where(dist_left < dist_right, idx_left, combined)
    combined = combined.to(torch.int32).view(weight_tensor.shape)
    base_idx = torch.div(combined, 256, rounding_mode='floor').to(torch.uint8)
    fine_idx = (combined % 256).to(torch.uint8)
    return base_idx, fine_idx
```

#### B) `PrimeQwenAttention(nn.Module)` — Drop-in Qwen3 Attention with Ghost Layers

Replace Qwen3's `q_proj`, `k_proj`, `v_proj`, `o_proj` with `GhostLinearTandem`.

- Preserve RoPE (import from transformers or re-implement)
- Preserve GQA (16 Q heads, 8 KV heads → repeat_kv logic stays)
- Preserve the causal attention mask

```
QWEN3 ATTN PROJECTIONS TO PRIMEIFY:
  q_proj:   (2048, 2048)  → GhostLinearTandem(2048, 2048)
  k_proj:   (2048, 1024)  → GhostLinearTandem(2048, 1024)  [GQA: 8 heads × 128]
  v_proj:   (2048, 1024)  → GhostLinearTandem(2048, 1024)
  o_proj:   (2048, 2048)  → GhostLinearTandem(2048, 2048)
```

#### C) `PrimeQwenMLP(nn.Module)` — Drop-in MLP with Ghost Layers

Qwen3 MLP uses SiLU gating: `output = (gate_proj(x) * silu(up_proj(x))) → down_proj`

```
QWEN3 MLP PROJECTIONS TO PRIMEIFY:
  gate_proj: (2048, 6144) → GhostLinearTandem(2048, 6144)
  up_proj:   (2048, 6144) → GhostLinearTandem(2048, 6144)
  down_proj: (6144, 2048) → GhostLinearTandem(6144, 2048)
```

Keep `F.silu` for the activation — do NOT quantize the activation function.

#### D) `PrimeQwenLayer(nn.Module)` — Full Decoder Block Wrapper

Wraps `PrimeQwenAttention + PrimeQwenMLP + RMSNorm`.
RMSNorm weights stay FP32 (standard), they are not GhostLinear.

#### E) `PrimeQwen3Model(nn.Module)` — Full Model Wrapper

- 28 × `PrimeQwenLayer` layers
- `TrueShadowlessEmbedding` for token embeddings (vocab=151,936, dim=2048)
- Standard RMSNorm for final norm
- `GhostLinearTandem(2048, 151936)` for the LM head

---

### 1.2 — `qwen3_prime_importer.py` (NEW — Weight Transplant Script)

This script runs ONCE to:

1. Load Qwen3-1.7B from HuggingFace (or local .safetensors)
2. Instantiate `PrimeQwen3Model` with fresh `GhostLinearTandem` layers
3. **For each layer**, call `fp16_weight_to_lut_index()` and copy the result into the
   `base_idx` / `fine_idx` buffers — seeding the voting engine with Qwen's knowledge
4. Scale factors initialized from `weight.abs().mean()` per output row
5. Save the initialized PRIME checkpoint as `qwen3_prime_init.pt`

```python
# Pseudo-code outline
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
prime_model = PrimeQwen3Model(config, lut=PRIMAL_LUT)

for layer_idx in range(28):
    qwen_layer = qwen_model.model.layers[layer_idx]
    prime_layer = prime_model.layers[layer_idx]

    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        qwen_w = qwen_layer.self_attn.__dict__[proj_name].weight.data
        prime_ghost = prime_layer.self_attn.__dict__[proj_name]
        base, fine = fp16_weight_to_lut_index(qwen_w, PRIMAL_LUT)
        prime_ghost.base_idx.copy_(base)
        prime_ghost.fine_idx.copy_(fine)
        # Initialize scale from weight magnitude
        prime_ghost.scale.data.copy_(qwen_w.abs().mean(dim=1, keepdim=True).float() * 2.0)

    # Repeat for MLP (gate, up, down)

torch.save(prime_model.state_dict(), "qwen3_prime_init.pt")
```

**CRITICAL NOTE on VRAM:**

- Qwen3-1.7B FP16 = ~3.4GB
- PRIME model `base_idx/fine_idx` (uint8×2) + scale (FP32) ≈ 2.1B params × 2 bytes ≈ ~4.2GB
- Importer script will need ≥8GB VRAM or run on CPU (slower but safe on 16GB RAM)
- **Recommendation:** Run importer on CPU first (`device='cpu'`), then move to GPU for training

---

### 1.3 — Update `manifolds.py` — Vocab-Scale Considerations

The existing `generate_int16_linear_manifold()` and `generate_int16_prime_manifold()`
generate a 65,536-entry LUT. This is **still correct** for Qwen3. No change needed.

However, the embedding table is now `(151,936 × 2048)` integers. At uint8×2:

- `base_idx`: 151936 × 2048 × 1 byte = **311 MB**
- `fine_idx`: 311 MB
- `vote_buffer` (int16): 151936 × 2048 × 2 bytes = **622 MB** ← **VRAM KILLER**

**Solution:** `TrueShadowlessEmbedding` vote buffer for this size should be:

- Stored on CPU and only moved to GPU in small batches during `apply_votes()`
- OR: Use `GhostLinearTandem` for the LM head, but keep token embeddings as **standard
  `nn.Embedding`** (FP16) and quantize it last, after the transformer layers are stable.
  This is **Phase 3 work** — don't primeify the embeddings first.

---

## PHASE 2: TRAINING LOOP ADAPTATION

### 2.1 — `qwen3_prime_train.py` (NEW — Trainer)

Fork from `primal_train_modular.py` with these key changes:

#### A) CONFIG Update

```python
CONFIG = {
    'vocab_size':  151936,  # Qwen3 tokenizer
    'seq_len':     512,     # Start conservative (32K is too big for GTX 1080 Ti)
    'dim':         2048,
    'n_layers':    28,
    'n_heads':     16,      # Q heads
    'n_kv_heads':  8,       # KV heads (GQA)
    'intermediate': 6144,
    'lr':          1e-4,
    'batch_size':  1,
    'grad_accum':  256,     # Larger accum to compensate for small batch
    'mode':        'primal',
    'device':      'cuda' if torch.cuda.is_available() else 'cpu'
}
```

#### B) Tokenizer Change

```python
# OUT: GPT2TokenizerFast.from_pretrained("gpt2")
# IN:
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
```

#### C) Dataset Strategy

The existing `TinyStories` loader is fine for Phase 1 stabilization. Later phases should
move to a higher-quality instruction set (Alpaca, OpenHermes, etc.) for fine-tuning.

#### D) Memory Budget (GTX 1080 Ti — 11GB VRAM)

```
Activations per layer (seq=512, dim=2048, batch=1):
  Hidden states: ~8MB per layer × 28 = 224MB
  Attention QKV: ~24MB per layer × 28 = 672MB
  MLP intermediates: ~48MB × 28 = 1344MB
  -----
  Total fwd pass: ~2.2GB
  Vote buffers (int16, 28 layers): 
    Attn: 4 projs × (2048×2048 or 2048×1024) × 2B × 28 layers ≈ 1.4GB
    MLP:  3 projs × (2048×6144) × 2B × 28 layers ≈ 2.1GB
  -----
  VRAM RISK: 3.5GB in vote buffers alone
```

**MITIGATION STRATEGIES:**

1. **Layer-wise vote decay:** After each `apply_tandem_votes()` call, vote buffers already
   decay by 0.95 (existing code). This is fine.
2. **Gradient checkpointing:** Enable `use_reentrant=False` on each `PrimeQwenLayer`.
3. **Mixed precision:** Use `torch.amp.autocast('cuda')` for forward pass — vote buffer
   writes remain in int16 (integer, not AMP-affected).
4. **Reduce seq_len to 256 initially** to save 4× activation memory.

#### E) The Sentinel

`AntigravitySentinelV571` works completely unchanged. It iterates `model.named_modules()`
and calls `apply_tandem_votes()` on every `GhostLinearTandem` it finds. Already perfect.

#### F) Night Shift Supervisor

Also works unchanged. However the **Master Lock protocol** needs a new target:

- OLD: Lock `model.head` until step 1000
- NEW: Lock the **LM head + ALL 28 layers** until step ~200, then unlock **2 layers per
  50 steps** (peeling from the top down) to prevent an avalanche.
  We call this **Protocol v7.00: LAYER PEEL**.

---

## PHASE 3: PROTOCOL v7.00 — LAYER PEEL

### The Problem

With 28 layers each containing 7 GhostLinearTandem projections, all vote buffers are
accumulating simultaneously from step 1. If we open all dams at step 1000 (like current
protocol), we risk a **28-layer synchronized avalanche** — massive flip events across
the entire network.

### The Solution: Progressive Layer Unlock

```
Steps   0 –  200: ALL layers frozen (vote buffers accumulate only, zero flips)
Steps 200 –  250: Layer 27 (top) unlocked — LM head flips allowed
Steps 250 –  300: Layer 26 unlocked
Steps 300 –  350: Layer 25 unlocked
...
Steps (200 + 27×50) = 1,550: Layer 0 (bottom) unlocked
Steps 1,550+: All layers active, normal Night Shift consensus tracking
```

This creates a **top-down crystallization cascade** — the output head crystallizes first
(like TinyLlama → PRIME did), then each layer beneath it gets its signal cleaned.

### Implementation

Add to `NightShiftSupervisor`:

```python
def get_unlocked_layers(self, current_step):
    if current_step < 200: return set()
    unlocked = set()
    for i in range(28):
        layer_idx = 27 - i  # Start from top
        unlock_step = 200 + (i * 50)
        if current_step >= unlock_step:
            unlocked.add(layer_idx)
    return unlocked
```

Modify `AntigravitySentinelV571.apply_safestep_and_vote()` to skip `apply_tandem_votes()`
on locked layers (set flips=0 for those modules).

---

## PHASE 4: VALIDATION & COMPRESSION

### 4.1 — Perplexity Benchmarks (reuse `primal_val_perplexity.py`)

- Update tokenizer and vocab_size
- Target: Perplexity < 20 after Phase 1 (500 steps), < 15 after Phase 2 (1000 steps)

### 4.2 — PRIME Packing (physical compression)

The existing `src/pack.py` (4-bit nibble packing) can be extended:

- Each weight in `GhostLinearTandem` is a 16-bit index (base×256 + fine) → 2 bytes
- These 16-bit indices can be further compressed with delta-encoding between adjacent
  positions (weights are correlated) → estimate 4-6 bits effective per weight
- A packed Qwen3-1.7B PRIME model: ~1.7B params × 2 bytes = 3.4GB → post-pack ≈ 600-800MB

### 4.3 — Inference Pipeline

`primal_infer_ghost.py` update:

1. Load `qwen3_prime_final.pt`
2. Generate PRIMAL_LUT on GPU
3. Forward pass: each `GhostLinearTandem.forward()` dequantizes on-the-fly
4. Qwen3 tokenizer for encode/decode
5. Uses Qwen3's native chat template (`<|im_start|>user\n...<|im_end|>`)

---

## IMPLEMENTATION CHECKLIST (Ordered)

```
[ ] PHASE 1: INFRASTRUCTURE
    [ ] 1.1  Write qwen3_prime_wrapper.py
              [ ] fp16_weight_to_lut_index()
              [ ] PrimeQwenAttention (with RoPE, GQA)
              [ ] PrimeQwenMLP (with SiLU gating)
              [ ] PrimeQwenLayer
              [ ] PrimeQwen3Model
    [ ] 1.2  Write qwen3_prime_importer.py
              [ ] Load Qwen3-1.7B weights from HuggingFace
              [ ] Seed GhostLinearTandem buffers with Qwen weights
              [ ] Verify LUT coverage (log % weights within LUT range before clamping)
              [ ] Save qwen3_prime_init.pt
    [ ] 1.3  Verify manifolds.py — no changes needed (65536 LUT is correct)

[ ] PHASE 2: TRAINING LOOP
    [ ] 2.1  Write qwen3_prime_train.py (fork of primal_train_modular.py)
              [ ] Update CONFIG (151936 vocab, dim=2048, 28 layers)
              [ ] Switch to Qwen3 tokenizer
              [ ] Add gradient checkpointing per layer
              [ ] Test single forward pass (VRAM check)
              [ ] Run 10-step smoke test, confirm vote buffers accumulating
              [ ] Run 100-step test, confirm any flips occurring

[ ] PHASE 3: PROTOCOL v7.00
    [ ] 3.1  Implement Layer Peel in NightShiftSupervisor
    [ ] 3.2  Update AntigravitySentinel to honor locked layer list
    [ ] 3.3  Run 500-step full training run
    [ ] 3.4  Benchmark perplexity after 500 steps

[ ] PHASE 4: VALIDATION
    [ ] 4.1  Full perplexity benchmark on WikiText-103
    [ ] 4.2  Interactive inference test with Qwen3 chat template
    [ ] 4.3  Pack and compress (target: < 800MB)
    [ ] 4.4  Update PAPER.md with Qwen3-1.7B results
```

---

## RISK REGISTER

| Risk | Severity | Mitigation |
|------|----------|------------|
| VRAM OOM during import | HIGH | Run importer on CPU |
| Vote buffers saturate too fast (large model) | HIGH | Start with `supermajority_threshold=30` (not 20) |
| Qwen weights outside LUT range [-1, 1] | MEDIUM | Apply `weight / weight.abs().max()` per-row normalization before snapping |
| GQA repeat_kv adds complexity | MEDIUM | Copy transformers' `repeat_interleave` logic verbatim |
| RoPE with theta=1M untested in PRIME | MEDIUM | Keep RoPE in FP32, outside vote buffer scope |
| Gradient flow breaks through 28 layers | MEDIUM | Gradient checkpointing + mild clipping (max_norm=0.5 initially) |
| Tokenizer vocab mismatch (151,936 tokens) | LOW | Importer handles it via `TrueShadowlessEmbedding(151936, 2048)` |

---

## QUICK-START COMMANDS (After Files Are Written)

```bash
# Step 1: Import Qwen3 weights into PRIME format
python qwen3_prime_importer.py --device cpu --output qwen3_prime_init.pt

# Step 2: Verify the checkpoint
python verify_modular_init.py --checkpoint qwen3_prime_init.pt

# Step 3: Start Phase 1 training
python qwen3_prime_train.py --manifold unified --resume --checkpoint qwen3_prime_init.pt

# Step 4: Monitor (existing tools work unchanged)
python monitor_primal.py
```

---

## WHY THIS WILL WORK

From PAPER.md: *"existing dense models possess a latent structure that can be 'evolved'
rather than retrained into a ternary state."*

Qwen3-1.7B has been trained on trillions of tokens. Its weight structure is maximally
efficient at its native precision. When we snap those weights to the nearest LUT coordinate:

- **Most weights are small** (bell-curve distribution near zero) → they land precisely
  near the prime-harmonic nucleus cluster in the LUT, which is densely packed there
- **Outlier weights** (large magnitude) → land on the tail section of the LUT (linear)
- **The scale parameter** per output row preserves the effective magnitude

The voting engine then refines each position from its FP16 starting coordinate toward
the nearest stable manifold point — guided by actual task loss gradients. This is
**Gradient-Evolved Quantization at 1.7B scale**.

The hard part is not the math. The hard part is VRAM management and preventing a
synchronized 28-layer avalanche. Protocol v7.00 Layer Peel solves that.

---
*End of Plan. Next action: Begin Phase 1 with `qwen3_prime_wrapper.py`.*
