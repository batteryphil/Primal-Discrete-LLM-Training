
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import manifolds

# ------------------------------------------------------------------------------
# 1. CONFIG & GRID (Must match training exactly)
# ------------------------------------------------------------------------------
CONFIG = {
    "dim": 768,
    "n_layers": 13, # Match Phase 53 training
    "n_heads": 12,
    "vocab_size": 50257,
    "seq_len": 512,
    "device": "cuda"
}

PRIMAL_LUT = manifolds.generate_manifold(device=CONFIG['device'])

# ------------------------------------------------------------------------------
# 2. GHOST COMPONENTS
# ------------------------------------------------------------------------------
class GhostLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_buffer('grid_indices', torch.zeros(out_features, in_features, dtype=torch.uint8))
        self.register_buffer('weight_buffer', torch.zeros(out_features, in_features, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(out_features, 1))

    def forward(self, x):
        # In inference, we just dequantize once (or use buffer)
        # We'll fill the buffer for consistency with the training architecture
        self.weight_buffer.copy_(PRIMAL_LUT[self.grid_indices.long()])
        self.weight_buffer.mul_(self.scale)
        return F.linear(x, self.weight_buffer)

class GhostBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c['dim'])
        self.attn = nn.MultiheadAttention(c['dim'], c['n_heads'], batch_first=True)
        self.ln2 = nn.LayerNorm(c['dim'])
        self.mlp = nn.Sequential(
            GhostLinear(c['dim'], 4 * c['dim']),
            nn.GELU(approximate='tanh'),
            GhostLinear(4 * c['dim'], c['dim'])
        )

    def forward(self, x, mask=None):
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class GhostGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.token_emb = nn.Embedding(c['vocab_size'], c['dim'])
        self.pos_emb = nn.Embedding(c['seq_len'], c['dim'])
        self.blocks = nn.ModuleList([GhostBlock(c) for _ in range(c['n_layers'])])
        self.ln_f = nn.LayerNorm(c['dim'])
        self.head = GhostLinear(c['dim'], c['vocab_size'])

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        return self.head(x)

    def generate(self, idx, max_new_tokens=20, temperature=0.8, top_k=40, repetition_penalty=1.2):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONFIG['seq_len']:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(idx.shape[1]):
                    token_id = idx[0, i].item()
                    # Only penalize if more than 1 occurrence? No, standard is to penalize always
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Optional: Top-K sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            
            if next_token.item() == 50256: 
                break
        return idx

# ------------------------------------------------------------------------------
# 3. LIVE INFERENCE
# ------------------------------------------------------------------------------
def chat():
    print(f"[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print(f"[*] Initializing GHOST 0.1B...")
    model = GhostGPT(CONFIG).to(CONFIG['device'])
    
    print(f"[*] Loading Live Checkpoint...")
    state_dict = torch.load("primal_ghost_live.pt")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("\n" + "="*50)
    print(" GHOST-CHAT: INTERACTIVE MODE")
    print(" (Type 'exit' to quit)")
    print("="*50)

    while True:
        try:
            prompt = input("\nUSER: ")
            if prompt.lower() in ["exit", "quit"]:
                break
            
            if not prompt.strip():
                continue
                
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(CONFIG['device'])
            
            print("GHOST: ", end="", flush=True)
            
            generated_ids = tokens
            with torch.no_grad():
                # Stream generation for feel
                for _ in range(50):
                    idx_cond = generated_ids[:, -CONFIG['seq_len']:]
                    logits = model(idx_cond)
                    logits = logits[:, -1, :]
                    
                    # Nucleus Sampling (Top-P) or simple Top-K for better coherence
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    generated_ids = torch.cat((generated_ids, next_token), dim=1)
                    word = tokenizer.decode([next_token.item()])
                    print(word, end="", flush=True)
                    
                    if next_token.item() == 50256:
                        break
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")

def infer(prompt="The sun is "):
    print(f"[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print(f"[*] Initializing GHOST 0.1B...")
    model = GhostGPT(CONFIG).to(CONFIG['device'])
    
    print(f"[*] Loading Live Checkpoint...")
    state_dict = torch.load("primal_ghost_live.pt")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"\nPROMPT: {prompt}")
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(CONFIG['device'])
    
    with torch.no_grad():
        print("GENERTING: ", end="", flush=True)
        tokens = model.generate(tokens, max_new_tokens=30)
        response = tokenizer.decode(tokens[0])
        print(response)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        infer(user_prompt)
    else:
        chat()
