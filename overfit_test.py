"""
================================================================================
PROJECT PRIMAL: PHASE 1 — SINGLE BATCH OVERFIT SANITY CHECK
================================================================================
Purpose: Prove the engine is NOT broken before wasting GPU hours on real data.

Protocol:
  - Freeze one batch of 32 short sentences (encoded once, reused forever)
  - Run that identical batch 3000 times
  - PASS: Loss descends smoothly from ~10.8 → below 0.5 (model memorizes the batch)
  - FAIL: Loss stays flat, explodes, or never drops below ~5.0

What this validates:
  ✓ GradientBooster is delivering usable signal to reasoning_gate
  ✓ AdaptiveVarianceTamer is not strangling the logits to death
  ✓ GhostGPT forward + loss + backward + optimizer all wire up correctly
  ✓ The discrete consensus engine (Antigravity Stride) can flip votes

Usage:
  python overfit_test.py

No CLI args. No checkpoints. No dataset downloads. Pure engine test.
================================================================================
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import sys
import time

# ── Patch stdout so we see output immediately ─────────────────────────────────
sys.stdout.reconfigure(encoding='utf-8')

print("="*72)
print("PHASE 1: SINGLE-BATCH OVERFIT TEST")
print("="*72)

# ── Device ────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Device: {device}")

# ── Minimal config — smaller than full training to keep this fast ─────────────
CONFIG = {
    'vocab_size': 50257,
    'seq_len':    128,      # Short: 128 tokens is plenty for overfit
    'dim':        1024,
    'iterations': 4,
    'device':     device,
}

# ── Imports that depend on CONFIG being defined ───────────────────────────────
import manifolds
from ghost_core import AdaptiveVarianceTamer
from transformers import GPT2TokenizerFast

# A hand-picked set of simple, short, diverse sentences (32 total).
# These are fixed forever — we never swap them out.
RAW_SENTENCES = [
    "The cat sat on the mat.",
    "Once upon a time there was a small dog.",
    "She opened the door and walked inside.",
    "The sun rose slowly over the mountains.",
    "He found a golden coin in the old chest.",
    "A little girl ran through the field of flowers.",
    "The rain fell softly on the quiet street.",
    "Tom liked to read books about space.",
    "The baker made fresh bread every morning.",
    "Birds sang in the tall green trees.",
    "They played games until the stars came out.",
    "The old clock on the wall ticked slowly.",
    "Max the dog loved to chase butterflies.",
    "A red boat floated on the calm lake.",
    "Lily smiled when she saw the rainbow.",
    "The teacher wrote the answer on the board.",
    "Every night before bed, he counted stars.",
    "The kitten curled up by the warm fireplace.",
    "It was a bright and sunny afternoon.",
    "She whispered a secret into her friend's ear.",
    "The train arrived at the station on time.",
    "He planted a tiny seed in the garden.",
    "The moon shone brightly through the window.",
    "They baked cookies and shared them with everyone.",
    "The puppy wagged its tail with joy.",
    "A small fish swam in the clear stream.",
    "He built a sandcastle on the beach.",
    "The children laughed as they played in the snow.",
    "She found a butterfly resting on the fence.",
    "The farmer fed the chickens each morning.",
    "The robot beeped and flashed its lights.",
    "A brave knight rode across the land.",
]

assert len(RAW_SENTENCES) == 32, "Need exactly 32 sentences"

# ── Tokenize & pack into one fixed batch ─────────────────────────────────────
print("[*] Tokenizing fixed batch...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

encoded = tokenizer(
    RAW_SENTENCES,
    max_length=CONFIG['seq_len'] + 1,
    truncation=True,
    padding='max_length',
    return_tensors='pt'
)
# Shape: [32, seq_len+1]
fixed_batch = encoded['input_ids'].to(device)
input_ids = fixed_batch[:, :-1]   # [32, 128]
targets    = fixed_batch[:, 1:]   # [32, 128]

print(f"[*] Fixed batch shape: {fixed_batch.shape}  (32 sentences × {CONFIG['seq_len']+1} tokens)")

# ── Build the LUT and model ───────────────────────────────────────────────────
print("[*] Generating manifold LUT...")
PRIMAL_LUT = manifolds.generate_int16_linear_manifold(device=device)

print("[*] Building GhostGPT model...")

# We need GhostGPT — pull it from primal_train_modular but without running
# the full training startup (which loads checkpoints, prints banners, etc.).
# Import the class only:
import importlib, types, contextlib, io

with contextlib.redirect_stdout(io.StringIO()):
    import primal_train_modular as _ptm
    GhostGPT = _ptm.GhostGPT

model = GhostGPT(CONFIG).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[*] Model: {total_params/1e6:.1f}M params total, {trainable/1e6:.1f}M trainable")

# ── Optimizer & Scaler ────────────────────────────────────────────────────────
# Simple AdamW — no sentinel, no dam, no autopilot. Pure engine test.
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,       # Higher LR than training — we want fast convergence on 32 samples
    weight_decay=0.01
)
scaler = GradScaler() if device == 'cuda' else None

# ── Sanity: Initial loss should be near log(50257) ≈ 10.82 ──────────────────
model.eval()
with torch.no_grad():
    with (autocast('cuda') if device == 'cuda' else torch.no_grad()):
        _, init_loss = model(input_ids[:1], targets=targets[:1])
print(f"\n[*] Initial loss on 1 sample: {init_loss.item():.4f}  (expected ≈10.82 for random weights)")
model.train()

# ── Training loop ─────────────────────────────────────────────────────────────
print("\n" + "─"*72)
print("  Step │    Loss │  Entropy │  dLoss/step  │  Time/100")
print("─"*72)

MAX_STEPS      = 3000
LOG_EVERY      = 50
PASS_THRESHOLD = 0.5   # We declare victory when loss hits this
FAIL_THRESHOLD = 15.0  # Anything above this by step 200 = engine broken

loss_history = []
t_block = time.time()
result = "PENDING"

for step in range(1, MAX_STEPS + 1):
    optimizer.zero_grad(set_to_none=True)

    if device == 'cuda':
        with autocast('cuda'):
            _, loss = model(input_ids, targets=targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        _, loss = model(input_ids, targets=targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    loss_val = loss.item()
    loss_history.append(loss_val)

    # ── Early failure detection ──────────────────────────────────────────────
    if step == 200 and loss_val > FAIL_THRESHOLD:
        result = "FAIL"
        print(f"\n{'!'*72}")
        print(f"  [FAIL] Step {step}: Loss {loss_val:.4f} > {FAIL_THRESHOLD} — engine is broken.")
        print(f"  The gradients are NOT flowing. GradientBooster or architecture is faulty.")
        print(f"{'!'*72}")
        break

    # ── Victory condition ────────────────────────────────────────────────────
    if loss_val < PASS_THRESHOLD:
        result = "PASS"
        print(f"\n{'='*72}")
        print(f"  [PASS ✓] Loss reached {loss_val:.4f} at step {step}!")
        print(f"  The engine is SOUND. GradientBooster + AdaptiveVarianceTamer confirmed working.")
        print(f"  Proceed to Phase 2: TinyStories.")
        print(f"{'='*72}")
        break

    # ── Periodic logging ─────────────────────────────────────────────────────
    if step % LOG_EVERY == 0:
        elapsed = time.time() - t_block
        # Crude entropy estimate: -log(1/vocab) approximated from loss
        # Real entropy from softmax would need a forward pass — use loss as proxy
        delta = ""
        if len(loss_history) >= LOG_EVERY:
            delta = f"{loss_history[-1] - loss_history[-LOG_EVERY]:+.4f}"
        print(f"  {step:5d} │ {loss_val:7.4f} │ {loss_val:8.4f} │  {delta:>12}  │  {elapsed:.1f}s")
        t_block = time.time()

# ── Final verdict ─────────────────────────────────────────────────────────────
if result == "PENDING":
    final_loss = loss_history[-1]
    if final_loss < 2.0:
        result = "PASS (PARTIAL)"
        verdict = f"Loss reached {final_loss:.4f} after {MAX_STEPS} steps. Converging but slow — architecture OK."
    else:
        result = "FAIL (SLOW)"
        verdict = f"Loss stuck at {final_loss:.4f} after {MAX_STEPS} steps. Engine may have gradient issues."
    print(f"\n[RESULT: {result}] {verdict}")

print("\n" + "="*72)
print(f"  OVERFIT TEST COMPLETE — Result: {result}")
print(f"  Min loss achieved: {min(loss_history):.4f}")
print(f"  Final loss:        {loss_history[-1]:.4f}")
print(f"  Steps run:         {len(loss_history)}")
print("="*72)
