import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import bitsandbytes as bnb
import gc
import os

# --- CONFIGURATION (PHASE 4: THE RE-EDUCATION) ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
CHECKPOINT_PATH = "trinity_evolved_500steps.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_STEPS = 200
GRAD_ACCUM = 8
MAX_LR = 2e-5  # Gentle re-education
MAX_SEQ_LEN = 512

# --- THE PRIME GRID ---
def get_prime_grid():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151]
    reciprocals = [1.0/p for p in primes]
    tails = [1.0, 1.5, 2.0, 3.0]
    full_grid = sorted([0.0] + reciprocals + tails + [-r for r in reciprocals] + [-t for t in tails])
    return torch.tensor(full_grid, device=DEVICE, dtype=torch.float16)

# --- QUANTIZATION (STE) ---
class PrimeQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, grid):
        sigma = weight.std()
        if sigma == 0: sigma = 1e-6
        w_norm = weight / sigma
        w_flat = w_norm.reshape(-1)
        w_quant_flat = torch.zeros_like(w_flat)
        chunk_size = 1024 * 1024
        grid_flat = grid.reshape(1, -1)
        for i in range(0, w_flat.numel(), chunk_size):
            chunk = w_flat[i : i + chunk_size].reshape(-1, 1)
            dist = torch.abs(chunk - grid_flat)
            nearest_idx = torch.argmin(dist, dim=1)
            w_quant_flat[i : i + chunk_size] = grid[nearest_idx]
        return w_quant_flat.view(weight.shape) * sigma
    @staticmethod
    def backward(ctx, grad_output): return grad_output, None

class PrimeLinear(nn.Linear):
    def __init__(self, original_layer):
        super().__init__(original_layer.in_features, original_layer.out_features, original_layer.bias is not None)
        with torch.no_grad():
            self.weight.data = original_layer.weight.data.clone()
            if original_layer.bias is not None: self.bias.data = original_layer.bias.data.clone()
        self.register_buffer('grid', get_prime_grid())
    def forward(self, x):
        w_quant = PrimeQuantFunction.apply(self.weight, self.grid)
        return F.linear(x, w_quant, self.bias)

def main():
    print("--- PROJECT TRINITY: PHASE 4 - THE RE-EDUCATION ---")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    model.gradient_checkpointing_enable()

    print("[INIT] Injecting Prime DNA...")
    replacements = []
    for name, module in model.named_modules():
         if isinstance(module, nn.Linear) and "head" not in name: replacements.append((name, module))
    for name, module in tqdm(replacements, desc="Swapping Layers"):
        parent = model.get_submodule(name.rsplit('.', 1)[0] if '.' in name else '')
        setattr(parent, name.rsplit('.', 1)[-1], PrimeLinear(module).to(DEVICE))

    if os.path.exists(CHECKPOINT_PATH):
        print(f"[INIT] Loading Checkpoint: {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH), strict=True)
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=MAX_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

    print("[DATA] Loading Alpaca Instruction Set...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)

    model.train()
    step = 0
    pbar = tqdm(total=TOTAL_STEPS)
    optimizer.zero_grad()

    for i, example in enumerate(dataset):
        if step >= TOTAL_STEPS: break
        
        # Alpaca Format
        prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse: "
        full_text = prompt + example['output'] + tokenizer.eos_token
        
        tokens = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(DEVICE)
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LEN).input_ids)
        
        labels = tokens.input_ids.clone()
        labels[:, :prompt_len] = -100 # Mask the instruction for re-education focus

        try:
            outputs = model(**tokens, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item()*GRAD_ACCUM:.4f}")
                
        except RuntimeError as e:
            if "out of memory" in str(e): torch.cuda.empty_cache()
            else: raise e

    print("\n[COMPLETE] Saving Trinity Final State...")
    torch.save(model.state_dict(), "trinity_final_reeducated.pt")

    # Final Verification
    model.eval()
    print("\n[VERIFY] Truthfulness Check:")
    test_prompt = "Instruction: What is the capital of France?\nResponse: "
    inp = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**inp, max_new_tokens=20, temperature=0.6, repetition_penalty=1.2)
    print(f"Result: {tokenizer.decode(out[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
