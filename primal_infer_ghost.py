
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# ------------------------------------------------------------------------------
# 1. CONFIG & GRID (Must match training exactly)
# ------------------------------------------------------------------------------
CONFIG = {
    "dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "vocab_size": 50257,
    "seq_len": 512,
    "device": "cuda"
}

LUT_VALS = [
    -1.0, -0.5, -0.333333, -0.2, -0.142857, -0.090909, -0.076923, 
    0.0, 
    0.076923, 0.090909, 0.142857, 0.2, 0.333333, 0.5, 1.0
]
PRIMAL_LUT = torch.tensor(LUT_VALS, device=CONFIG['device'])

# ------------------------------------------------------------------------------
# 2. GHOST COMPONENTS
# ------------------------------------------------------------------------------
class GhostLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_buffer('grid_indices', torch.zeros(out_features, in_features, dtype=torch.uint8))
        self.register_buffer('weight_buffer', torch.zeros(out_features, in_features, dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(1.0))

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

# ------------------------------------------------------------------------------
# 3. LIVE INFERENCE
# ------------------------------------------------------------------------------
def infer(prompt="The sun is "):
    print(f"[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print(f"[*] Initializing GHOST 0.1B...")
    model = GhostGPT(CONFIG).to(CONFIG['device'])
    
    print(f"[*] Loading Live Checkpoint...")
    state_dict = torch.load("primal_ghost_live.pt")
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"\nPROMPT: {prompt}")
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(CONFIG['device'])
    
    with torch.no_grad():
        logits = model(tokens)
        next_token_logits = logits[0, -1, :]
        
        # Get top 5
        probs = F.softmax(next_token_logits, dim=-1)
        top_k = torch.topk(probs, 5)
        
        print("\nTOP PREDICTIONS:")
        for i in range(5):
            token_id = top_k.indices[i].item()
            prob = top_k.values[i].item()
            word = tokenizer.decode([token_id])
            print(f"{i+1}. '{word}' (Prob: {prob:.4f})")

if __name__ == "__main__":
    import sys
    user_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The sun is "
    infer(user_prompt)
