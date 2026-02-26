import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from qwen25_prime_wrapper import PrimeQwen2CausalLM

import sys

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
sys.stdout = LoggerTee("training_qwen.log")

# ==============================================================================
# NIGHT SHIFT SUPERVISOR V7: THE LAYER PEEL
# ==============================================================================
class NightShiftSupervisorV7:
    def __init__(self, model, optimizer, base_lr=1e-4):
        self.model = model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_layers = 28
        self.locked_layers = set(range(self.total_layers))
        
        # Initial State: Absolute Lockdown
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            
        # The LM Head and Embeddings are allowed to thaw early
        print("[NIGHT SHIFT] Initializing with Head & Embedding thawing...")
        for param in self.model.lm_head.parameters():
            param.requires_grad = True
        for param in self.model.embed_tokens.parameters():
            param.requires_grad = True

    def step_peel(self, current_step):
        """Protocol v7.00: Top-Down Crystallization Cascade"""
        if current_step < 200:
            return  # Head-only stabilization phase

        # Every 50 steps after 200, we peel back 2 layers.
        peel_index = (current_step - 200) // 50
        layers_to_thaw_count = peel_index * 2
        
        target_unlocked_count = min(layers_to_thaw_count, self.total_layers)
        current_unlocked_count = self.total_layers - len(self.locked_layers)
        
        if target_unlocked_count > current_unlocked_count:
            to_unlock = target_unlocked_count - current_unlocked_count
            for _ in range(to_unlock):
                if not self.locked_layers: break
                layer_idx = max(self.locked_layers) 
                print(f"[NIGHT SHIFT] Step {current_step}: Unlocking Layer {layer_idx}...")
                layer = self.model.layers[layer_idx]
                for param in layer.parameters():
                    param.requires_grad = True
                self.locked_layers.remove(layer_idx)
                
            # [CONTINGENCY] The Autograd Expansion Threat
            if current_step == 200:
                print(f"\n[NIGHT SHIFT] [CRITICAL] Initiating VRAM Contigency Protocol for Step 200...")
                # We do this globally in train() now, we just log it here.
                pass

# ==============================================================================
# DATA LOADING (Answer-Masked SFT)
# ==============================================================================
import threading
import queue

def get_dataloader(tokenizer, seq_len=512, prefetch_size=10):
    print("[DATA] Loading Python Instructions for Accuracy Recovery (Masked SFT)...")
    dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    
    # Assistant Start Token Sequence for Qwen2.5: <|im_start|>assistant\n
    # Usually: [151644, 77091, 198] or similar
    
    def gen_batches():
        buffer_ids = []
        buffer_labels = []
        
        for item in dataset:
            prompt = f"<|im_start|>user\n{item['instruction']}\n{item['input']}<|im_end|>\n"
            response = f"<|im_start|>assistant\n{item['output']}<|im_end|>"
            
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            
            # Pack sequences continuously
            buffer_ids.extend((prompt_ids + response_ids))
            buffer_labels.extend(([-100] * len(prompt_ids) + response_ids))
            
            # When we have enough for a perfect sequence block, yield it
            while len(buffer_ids) >= seq_len:
                out_ids = buffer_ids[:seq_len]
                out_labels = buffer_labels[:seq_len]
                
                buffer_ids = buffer_ids[seq_len:]
                buffer_labels = buffer_labels[seq_len:]
                
                yield torch.tensor([out_ids]).pin_memory(), torch.tensor([out_labels]).pin_memory()
            
    def fetch_worker(generator, q_out):
        try:
            for item in generator():
                q_out.put(item)
        except Exception:
            pass
        finally:
            q_out.put(None)

    q = queue.Queue(maxsize=prefetch_size)
    th = threading.Thread(target=fetch_worker, args=(gen_batches, q), daemon=True)
    th.start()
    
    def dataloader():
        while True:
            item = q.get()
            if item is None: break
            yield item
            
    return dataloader()

# ==============================================================================
# TRAINING ENGINE
# ==============================================================================
def train():
    CONFIG = {
        'hidden_size': 1536,
        'num_hidden_layers': 28,
        'num_attention_heads': 12,
        'num_key_value_heads': 2,
        'intermediate_size': 8960,
        'vocab_size': 151936,
        'lr': 1e-4,
        'grad_accum': 256,
        'seq_len': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    from manifolds import generate_int16_prime_manifold
    lut = generate_int16_prime_manifold(device=CONFIG['device'])

    model = PrimeQwen2CausalLM(CONFIG, lut=lut)
    
    # [RESUME LOGIC] Dynamically find the latest checkpoint
    import glob
    checkpoints = glob.glob("qwen25_coder_prime_step_*.pt")
    latest_step = 0
    checkout_point = "qwen25_coder_prime_init.pt"
    
    for ckpt in checkpoints:
        if "resume" in ckpt: continue
        try:
            step = int(ckpt.split("_step_")[1].split(".")[0])
            if step > latest_step:
                latest_step = step
                checkout_point = ckpt
        except: pass
        
    if latest_step > 0:
        print(f"[*] Resuming from checkpoint: {checkout_point}")
        state_dict = torch.load(checkout_point, map_location='cpu')
    else:
        print(f"[*] Loading mapped weights: qwen25_coder_prime_init.pt")
        state_dict = torch.load("qwen25_coder_prime_init.pt", map_location='cpu')
        
    model.load_state_dict(state_dict)
    del state_dict
    
    # [MISSION OVERRIDE] Unfreeze Layer 26 to provide context to the starved Layer 27
    print("[!] MISSION OVERRIDE: Layer 26 Unfrozen. Gradient Flow Expanded.")
    for param in model.layers[26].parameters():
        param.requires_grad = True
        
    model = model.to(CONFIG['device'])
    torch.cuda.empty_cache()
    
    for layer in model.layers:
        layer.gradient_checkpointing = True

    try:
        print("[*] Engaging PyTorch 2.0 Compiler...")
        # Torch compile often fails gracefully on unsupported platforms, but if it works it's a huge speedup.
        import sys
        if sys.platform != "win32": # It crashes unrecoverably deep in inductor on standard Windows 
            model = torch.compile(model, mode="max-autotune")
            print("    -> Compilation requested successfully.")
        else:
            print("    -> Compilation skipped (unsupported on native Windows MSVC build without WSL).")
    except Exception as e:
        print(f"[!] Torch compile skipped: {e}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    supervisor = NightShiftSupervisorV7(model, optimizer, base_lr=CONFIG['lr'])
    
    dataloader = get_dataloader(tokenizer, seq_len=CONFIG['seq_len'])
    
    current_step = latest_step if latest_step > 0 else 0
    # REDUCED for health check and faster dashboard feedback
    accum_steps = 32 # Original was 256
    seq_len = CONFIG['seq_len']
    
    print(f"[*] Starting Accuracy Recovery (Step size: {accum_steps} batches)...")
    model.train()
    
    total_loss = 0
    start_time = time.time()
    batch_start_time = time.time()
    
    import signal
    def emergency_save(*args):
        print(f"\n\n[!] EMERGENCY ABORT INITIATED. Saving progress at Step {current_step}...")
        torch.save(model.state_dict(), f"qwen25_coder_prime_step_{current_step}_resume.pt")
        print(f"[!] Saved: qwen25_coder_prime_step_{current_step}_resume.pt. Exiting gracefully.")
        sys.exit(0)
    signal.signal(signal.SIGINT, emergency_save)
    
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        # [CONTINGENCY] The Grand Synchronization / Sealing Pass
        if current_step >= 900:
            if seq_len != 128:
                print("\n[NIGHT SHIFT] [SEALING HYPER-DRIVE] Activating Global Desync Recovery...")
                seq_len = 128
                accum_steps = int(accum_steps * (384/128)) * 2 # Increase batch size by another 2x to smooth gradients
                print(f"[*] New seq_len: {seq_len} | New accum_steps: {accum_steps}")
                
                # Unfreeze EVERYTHING for the final sync
                for name, param in model.named_parameters():
                    param.requires_grad = True
                    
        # [CONTINGENCY] VRAM Tightening at Step 200 Activation
        elif current_step >= 200 and seq_len == 512:
            print("\n[VRAM CONTINGENCY] Reducing seq_len and increasing accum_steps...")
            seq_len = 384
            accum_steps = int(accum_steps * (512/384))
            print(f"[*] New seq_len: {seq_len} | New accum_steps: {accum_steps}")
            
        # Re-slice inputs in case the dataloader yields the original 512 limit
        input_ids = input_ids[:, :seq_len].to(CONFIG['device'])
        labels = labels[:, :seq_len].to(CONFIG['device'])
        
        logits, _ = model(input_ids)
        
        # Cross Entropy ignoring -100 (Answer-Masked)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, CONFIG['vocab_size']), shift_labels.view(-1))
        
        loss_val = loss.item()
        loss = loss / accum_steps
        loss.backward()
        total_loss += loss_val / accum_steps
        
        batch_duration = time.time() - batch_start_time
        batch_tps = CONFIG['seq_len'] / batch_duration
        
        print(".", end="", flush=True)
        if (batch_idx + 1) % 10 == 0:
            print(f"[{batch_idx+1}/{accum_steps}] L:{loss_val:.4f} TPS:{batch_tps:.1f}", end=" ", flush=True)
        
        batch_start_time = time.time()
        
        if (batch_idx + 1) % accum_steps == 0:
            current_step += 1
            print(f"\n[SYNCHRONIZE] Step {current_step} finalizing...")
            
            supervisor.step_peel(current_step)
            
            flips = 0
            for name, module in model.named_modules():
                if hasattr(module, 'apply_tandem_votes'):
                    if any(p.requires_grad for p in module.parameters()):
                        res = module.apply_tandem_votes(learning_rate=CONFIG['lr'])
                        if isinstance(res, dict): flips += res['flips']
                elif hasattr(module, 'apply_votes'):
                    if any(p.requires_grad for p in module.parameters()):
                        flips += module.apply_votes(consensus_threshold=5)
            
            optimizer.step()
            optimizer.zero_grad()
            
            elapsed = time.time() - start_time
            tps = (CONFIG['seq_len'] * accum_steps) / elapsed
            
            stats = {
                "step": current_step,
                "loss": round(total_loss, 4),
                "flips": round(flips / 1e6, 4),
                "tps": round(tps, 2),
                "lr": CONFIG['lr'],
                "timestamp": time.time()
            }
            
            try:
                history = []
                if os.path.exists("stats_coder.json"):
                    with open("stats_coder.json", "r") as f:
                        history = json.load(f)
                        # Ensure we don't have overlapping future steps from a previous failed run
                        history = [h for h in history if h.get("step", 0) < current_step]
                history.append(stats)
                with open("stats_coder.json", "w") as f:
                    json.dump(history[-500:], f)
            except: pass
            
            log_line = f"Step {current_step} | Loss: {total_loss:.4f} | Flips: {flips} | TPS: {tps:.2f} | Time: {elapsed:.2f}s"
            print(f"\n{log_line}")
                
            total_loss = 0
            start_time = time.time()
            
            # [WORD SALAD] Autonomous text generation to gauge structural healing
            if current_step % 10 == 0 or current_step == 1:
                print(f"\n[*] Generating Word Salad (Assessing Perplexity: {torch.exp(loss).item():.2f})...")
                try:
                    model.eval()
                    with torch.no_grad():
                        sample_prompt = "<|im_start|>user\nWrite a python function to print hello world\n<|im_end|>\n<|im_start|>assistant\n"
                        # For extremely constrained VRAM, we slice just the immediate prompt
                        inputs = tokenizer(sample_prompt, return_tensors="pt").to(CONFIG['device'])
                        input_ids = inputs["input_ids"]
                        
                        # Manual Token Sampling Loop with Penalties
                        generated_ids = []
                        temperature = 0.7
                        top_p = 0.8
                        repetition_penalty = 1.5
                        
                        for _ in range(60):
                            logits, _ = model(input_ids)
                            next_token_logits = logits[:, -1, :]
                            
                            # Hard ban on FIM / unwanted special tokens showing up in word salad
                            for bad_tok in [151659, 151660, 151661, 151662, 151663, 151664]:
                                next_token_logits[0, bad_tok] = -float('inf')
                                
                            # 1. Repetition Penalty
                            if repetition_penalty != 1.0 and len(generated_ids) > 0:
                                for token_id in set(generated_ids):
                                    if next_token_logits[0, token_id] < 0:
                                        next_token_logits[0, token_id] *= repetition_penalty
                                    else:
                                        next_token_logits[0, token_id] /= repetition_penalty
                                        
                            # 1.5 Strict loop prevention (Ban repeating the exact same token 3 times sequentially)
                            if len(generated_ids) >= 2 and generated_ids[-1] == generated_ids[-2]:
                                next_token_logits[0, generated_ids[-1]] = -float('inf')
                                        
                            # 2. Temperature Tuning
                            next_token_logits = next_token_logits / temperature
                            
                            # 3. Top-P (Nucleus) Sampling
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            next_token_logits[indices_to_remove] = float('-inf')
                            
                            # 4. Final Sampling Selection
                            probs = F.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                            
                            generated_ids.append(next_token.item())
                            input_ids = torch.cat([input_ids, next_token], dim=-1)
                            
                            if next_token.item() in [tokenizer.eos_token_id, 151645]: # 151645 is likely <|im_end|>
                                break
                                
                        salad = tokenizer.decode(generated_ids, skip_special_tokens=False)
                        
                        # Clean up formatting for dashboard
                        clean_salad = salad.replace("<|im_start|>", "").replace("<|im_end|>", "")
                        
                        # Save to API source file
                        salad_history = []
                        if os.path.exists("samples_coder.json"):
                            with open("samples_coder.json", "r") as f:
                                salad_history = json.load(f)
                        salad_history.append({"step": current_step, "text": clean_salad})
                        with open("samples_coder.json", "w") as f:
                            json.dump(salad_history[-10:], f)
                            
                        print(f"    -> Output saved to dashboard.")
                except Exception as e:
                    print(f"    -> Generator skipped (VRAM limitation): {e}")
                model.train()
                
            if current_step % 100 == 0:
                torch.save(model.state_dict(), f"qwen25_coder_prime_step_{current_step}.pt")

if __name__ == "__main__":
    train()
