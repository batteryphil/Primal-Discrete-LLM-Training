"""
Trinity Comprehensive Benchmark Suite (Step 12172)
===================================================
Runs a full diagnostic battery on the current checkpoint:
  1. WikiText-2 Perplexity (GPU)
  2. Trinity Density Analysis (LUT utilization)
  3. Salad Test (Text Generation Quality)
  4. Weight Distribution Analysis
  5. Scale Health Check
"""
import torch
import torch.nn.functional as F
import time
import sys
import os
import json

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 60)
print("  TRINITY BENCHMARK SUITE")
print("=" * 60)

# --- SETUP ---
t0 = time.time()
from primal_train_ghost import GhostGPT, CONFIG
print(f"[*] Import took {time.time()-t0:.1f}s")

device = CONFIG['device']
model = GhostGPT(CONFIG).to(device)
sd = torch.load("primal_ghost_live.pt", map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()
print(f"[*] Model loaded on {device} in {time.time()-t0:.1f}s")

results = {}

# =====================================================
# BENCHMARK 1: Trinity Density Analysis
# =====================================================
print("\n" + "=" * 60)
print("  BENCHMARK 1: TRINITY DENSITY ANALYSIS")
print("=" * 60)

active_levels = set()
layer_stats = {}
for name, m in model.named_modules():
    if hasattr(m, 'grid_indices'):
        unique = m.grid_indices.unique().tolist()
        active_levels.update(unique)
        layer_stats[name] = len(unique)

total_active = len(active_levels)
eff_bits = torch.log2(torch.tensor(float(total_active))).item()

print(f"  Active Levels:    {total_active} / 256")
print(f"  Effective Bits:   {eff_bits:.2f}")
print(f"  Layer Breakdown:")
for name, count in sorted(layer_stats.items(), key=lambda x: x[1]):
    bar = "█" * (count // 5)
    print(f"    {name:40s} {count:3d}/256  {bar}")

results['density'] = {
    'active_levels': total_active,
    'effective_bits': round(eff_bits, 3),
    'layers': layer_stats
}

# =====================================================
# BENCHMARK 2: Scale Health Check
# =====================================================
print("\n" + "=" * 60)
print("  BENCHMARK 2: SCALE HEALTH CHECK")
print("=" * 60)

scales = []
scale_details = {}
for name, m in model.named_modules():
    if hasattr(m, 'scale'):
        s = m.scale.detach().float()
        mean_s = s.mean().item()
        std_s = s.std().item()
        min_s = s.min().item()
        max_s = s.max().item()
        scales.append(mean_s)
        scale_details[name] = {
            'mean': round(mean_s, 6),
            'std': round(std_s, 6),
            'min': round(min_s, 6),
            'max': round(max_s, 6)
        }

if scales:
    avg_scale = sum(scales) / len(scales)
    print(f"  Average Ghost Scale: {avg_scale:.6f} (Target: 1.0)")
    print(f"  Scale Range:         [{min(scales):.4f}, {max(scales):.4f}]")
    print(f"  Layers Tracked:      {len(scales)}")
    
    # Flag unhealthy scales
    unhealthy = [(n, d) for n, d in scale_details.items() 
                 if d['mean'] < 0.9 or d['mean'] > 1.1]
    if unhealthy:
        print(f"  [!] WARNING: {len(unhealthy)} layers with scale drift > 10%")
        for name, detail in unhealthy:
            print(f"      {name}: mean={detail['mean']}")
    else:
        print(f"  [OK] All scales within 10% of unity.")
    
    results['scales'] = {
        'average': round(avg_scale, 6),
        'range': [round(min(scales), 4), round(max(scales), 4)],
        'unhealthy_count': len(unhealthy)
    }

# =====================================================
# BENCHMARK 3: Weight Distribution Analysis
# =====================================================
print("\n" + "=" * 60)
print("  BENCHMARK 3: WEIGHT DISTRIBUTION ANALYSIS")
print("=" * 60)

from manifolds import generate_manifold
manifold = generate_manifold()

all_indices = []
for m in model.modules():
    if hasattr(m, 'grid_indices'):
        all_indices.append(m.grid_indices.flatten())

if all_indices:
    all_idx = torch.cat(all_indices)
    total_params = all_idx.numel()
    
    # Count frequency of each index
    counts = torch.bincount(all_idx.cpu(), minlength=256)
    
    # Top 10 most used values
    top_k = 10
    top_counts, top_indices = counts.sort(descending=True)
    
    print(f"  Total Grid Parameters: {total_params:,}")
    print(f"  Top {top_k} Most-Used Manifold Values:")
    for i in range(top_k):
        idx = top_indices[i].item()
        cnt = top_counts[i].item()
        pct = cnt / total_params * 100
        val = manifold[idx].item() if idx < len(manifold) else "?"
        print(f"    #{i+1:2d}  Index {idx:3d} = {val:+.4f}  ({cnt:>8,} params, {pct:.1f}%)")
    
    # Concentration: what % of params are in top-3 values?
    top3_pct = top_counts[:3].sum().item() / total_params * 100
    top10_pct = top_counts[:10].sum().item() / total_params * 100
    
    print(f"\n  Concentration:")
    print(f"    Top  3 values hold: {top3_pct:.1f}% of all weights")
    print(f"    Top 10 values hold: {top10_pct:.1f}% of all weights")
    
    # Check for ternary convergence (-1, 0, +1)
    # Find manifold indices closest to -1, 0, +1
    ternary_pct = 0
    for target in [-1.0, 0.0, 1.0]:
        diffs = (manifold - target).abs()
        closest_idx = diffs.argmin().item()
        ternary_pct += counts[closest_idx].item()
    ternary_pct = ternary_pct / total_params * 100
    
    print(f"    Ternary ({-1},0,{1}) hold: {ternary_pct:.1f}% of all weights")
    
    results['weight_dist'] = {
        'total_params': total_params,
        'top3_concentration': round(top3_pct, 1),
        'top10_concentration': round(top10_pct, 1),
        'ternary_concentration': round(ternary_pct, 1)
    }

# =====================================================
# BENCHMARK 4: Salad Test (Text Generation)
# =====================================================
print("\n" + "=" * 60)
print("  BENCHMARK 4: SALAD TEST (Text Generation)")
print("=" * 60)

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

prompts = [
    "The meaning of life is",
    "Once upon a time",
    "In the year 2025",
    "The capital of France is",
    "def fibonacci(n):",
]

gen_results = []
for prompt in prompts:
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    
    generated = list(tokens)
    with torch.no_grad():
        for _ in range(30):
            x = torch.tensor([generated[-CONFIG['seq_len']:]],
                           device=device)
            logits, _ = model(x)
            next_logits = logits[0, -1, :]
            # Temperature sampling
            probs = F.softmax(next_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
    
    output = tokenizer.decode(generated)
    gen_results.append({'prompt': prompt, 'output': output})
    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Output: {output[:120]}...")

results['salad_test'] = gen_results

# =====================================================
# BENCHMARK 5: WikiText-2 Perplexity (GPU)
# =====================================================
print("\n" + "=" * 60)
print("  BENCHMARK 5: WIKITEXT-2 PERPLEXITY (GPU)")
print("=" * 60)

from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])
encodings = tokenizer(text, return_tensors="pt")
input_ids_full = encodings.input_ids[0]

seq_len = CONFIG['seq_len']
total_tokens = input_ids_full.size(0)
num_batches = total_tokens // seq_len

print(f"  Evaluating over {total_tokens:,} tokens ({num_batches} batches)...")

total_loss = 0.0
total_count = 0
t_start = time.time()

with torch.no_grad():
    for i in range(num_batches):
        batch = input_ids_full[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
        logits, _ = model(batch)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                              shift_labels.view(-1))
        total_loss += loss.item()
        total_count += 1
        
        if (i + 1) % 100 == 0:
            avg = total_loss / total_count
            print(f"  [{i+1}/{num_batches}] Running Avg Loss: {avg:.4f}", flush=True)

avg_loss = total_loss / total_count
perplexity = torch.exp(torch.tensor(avg_loss)).item()
eval_time = time.time() - t_start

print(f"\n  Final Average Loss: {avg_loss:.4f}")
print(f"  VALIDATION PERPLEXITY: {perplexity:.2f}")
print(f"  Evaluation Time: {eval_time:.1f}s ({total_count/eval_time:.1f} batches/sec)")

results['perplexity'] = {
    'value': round(perplexity, 2),
    'avg_loss': round(avg_loss, 4),
    'eval_time_s': round(eval_time, 1)
}

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "=" * 60)
print("  FINAL SUMMARY — Trinity @ Step 12172")
print("=" * 60)
print(f"  Perplexity:        {results['perplexity']['value']:.2f}")
print(f"  Density:           {results['density']['active_levels']}/256 ({results['density']['effective_bits']:.2f} eff. bits)")
if 'scales' in results:
    print(f"  Scale Health:      {'HEALTHY' if results['scales']['unhealthy_count'] == 0 else 'WARNING'} (avg={results['scales']['average']:.4f})")
if 'weight_dist' in results:
    print(f"  Top-3 Concentration: {results['weight_dist']['top3_concentration']}%")
    print(f"  Ternary Weight:    {results['weight_dist']['ternary_concentration']}%")
print(f"  Salad Tests:       {len(gen_results)} prompts evaluated")
print("=" * 60)

# Save results
with open("benchmark_results.json", "w") as f:
    # Make gen_results serializable
    json.dump(results, f, indent=2, default=str)
print("[*] Results saved to benchmark_results.json")
