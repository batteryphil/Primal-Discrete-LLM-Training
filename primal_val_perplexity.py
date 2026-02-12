import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from primal_train_ghost import GhostGPT, CONFIG, GhostLinear
import os
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def calculate_perplexity():
    print("[*] Loading Model Architecture...", flush=True)
    # Ensure config matches training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GhostGPT(CONFIG).to(device)
    
    checkpoint_path = "primal_ghost_live.pt"
    if not os.path.exists(checkpoint_path):
        print(f"[!] Checkpoint {checkpoint_path} not found!", flush=True)
        return

    print(f"[*] Loading Checkpoint {checkpoint_path}...", flush=True)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("[*] Loading Validation Dataset (WikiText-2)...", flush=True)
    try:
        # Use WikiText-2 Test Split for Standardized Perplexity
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"[!] Failed to load WikiText-2: {e}")
        return

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    max_length = CONFIG['seq_len']
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    print(f"[*] Evaluating over {seq_len} tokens...", flush=True)
    
    count = 0
    total_steps = seq_len // stride
    
    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            
            if input_ids.size(1) < 2:
                continue
                
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            # Forward Pass
            # GhostGPT forward returns (logits, loss)
            # But the loss inside GhostGPT runs on targets if provided.
            # We can use the internal loss calculation.
            
            # Since inputs/targets need to be aligned for the model:
            # model(idx, targets)
            # But wait, GhostGPT.forward expects `targets` to be shifted?
            # Let's check GhostGPT.forward signature in primal_train_ghost.py imports
            # It takes `targets`. In training: `logits, loss = model(input_ids, targets=targets)`
            # And `input_ids` is `batch[:, :-1]`, `targets` is `batch[:, 1:]`.
            # So the model computes loss internally.
            
            # Here for sliding window, we need to be careful.
            # Let's just compute loss manually to be safe and standard.
            # Or pass `input_ids` and let it return loss.
            # But we need specific targets.
            
            # Let's assume standard behavior:
            # Input: tokens [0..N]
            # Target: tokens [1..N+1]
            
            # Prepare inputs for the model
            inp = input_ids[:, :-1]
            tgt = input_ids[:, 1:]
            
            logits, _ = model(inp)
            
            # Shift logits and labels (already shifted by logic above)
            # Logits: [B, T, V]
            # Targets: [B, T]
            
            shift_logits = logits # already valid for next token
            shift_labels = tgt
            
            # Only compute loss for the target window (stride)
            # Evaluating perplexity with sliding window usually means we mask out the context we've already seen?
            # For simplicity, let's just use standard non-sliding or naive sliding.
            # Stride = Max Length -> Non-overlapping.
            # Let's use Non-overlapping chunks of 512 for speed and simplicity.
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            nlls.append(loss)
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
                
            count += 1
            if count % 10 == 0:
                print(f"[{count}/{total_steps}] Batch Loss: {loss.item():.4f}", end="\r", flush=True)

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\n\n[*] Validation Perplexity: {ppl.item():.4f}")
    
    # Primal Telemetry
    print("-" * 40)
    print("PRIMAL TELEMETRY")
    print("-" * 40)
    primal_scales = [m.scale for m in model.modules() if isinstance(m, GhostLinear)]
    if primal_scales:
        avg_scale = torch.stack(primal_scales).mean().item()
        print(f"Average Ghost Scale: {avg_scale:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    calculate_perplexity()
