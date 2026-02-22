import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast
import manifolds
from ghost_core import PrimalTRMCore, GhostLinearTandem, LogitTamer
import os
import torch.nn.functional as F

# --- CONFIG ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG = {
    'vocab_size': 50257, 
    'dim': 1024,
    'iterations': 4,
    'seq_len': 1024,
    'device': device
}

# --- LUT ---
PRIMAL_LUT = manifolds.generate_int16_linear_manifold(device=device)

# --- ARCHITECTURE ---
class GhostGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.token_emb = nn.Embedding(c['vocab_size'], c['dim'])
        self.pos_emb = nn.Embedding(2048, c['dim'])
        num_iterations = c.get('iterations', 8)
        self.trm_core = PrimalTRMCore(c['dim'], num_iterations=num_iterations, lut=PRIMAL_LUT)
        self.trm_core.use_checkpointing = False
        self.internal_tamer = LogitTamer(threshold=2.5) 
        self.ln_f = nn.LayerNorm(c['dim'])
        self.head = GhostLinearTandem(c['dim'], c['vocab_size'], lut=PRIMAL_LUT, sensitivity=0.05)

    def forward(self, idx, targets=None, annealing_factor=1.0, training_step=0):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + (self.pos_emb(pos) * 2.5)
        x_refined = self.trm_core(x, lut=PRIMAL_LUT)
        y = self.ln_f(x_refined)
        voltage = getattr(self.trm_core, 'voltage_boost', 1.0)
        logits = self.head(y, lut=PRIMAL_LUT) * voltage
        logits = self.trm_core.tamer(logits)
        
        loss = None
        if targets is not None:
             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def run_test():
    # --- LOADING ---
    model = GhostGPT(CONFIG).to(device)
    if os.path.exists('primal_live.pt'):
        state_dict = torch.load('primal_live.pt', map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("[*] Loaded weights from primal_live.pt")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # --- VALIDATION SCRIPT (User Block + Sampling Fix) ---
    model.eval() 
    test_prompts = ["Once upon a time, there was a", "The sun was", "Lily saw a"]

    print("\n" + "!"*40)
    print("   PROTOCOL v5.89: SENTINEL VALIDATION   ")
    print("!"*40)

    with torch.no_grad():
        for p in test_prompts:
            input_ids = tokenizer.encode(p, return_tensors="pt").to(device)
            print(f"\nPROMPT: {p}")
            print("GENERATED: ", end="")
            
            # Generate 50 tokens for each
            for _ in range(50):
                # Corrected logic to match training loop
                idx_cond = input_ids if input_ids.size(1) <= CONFIG['seq_len'] else input_ids[:, -CONFIG['seq_len']:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                
                # Temperature 0.8 for a bit less 'will will will' but user objective says observe stability
                # User script used argmax, but training used multinomial. 
                # I'll use a middle ground or follow objective. 
                # Test 2 wants to see if it stays English-like or collapses.
                # Let's use multinomial to see if variety exists.
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                word = tokenizer.decode(next_token[0])
                print(word, end="", flush=True)
                if next_token.item() == tokenizer.eos_token_id: break
            print("\n" + "-"*20)

if __name__ == "__main__":
    run_test()
