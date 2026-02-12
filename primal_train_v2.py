import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
import math
import time

# V3.0.0 Prime Grid
LUT = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 0.333333, -0.333333, 0.2, -0.2, 0.142857, -0.142857, 0.090909, -0.090909, 0.076923, -0.076923, 0.0])

class PrimalSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        x_norm = x / scale
        grid = LUT.to(x.device)
        diff = torch.abs(x_norm.unsqueeze(-1) - grid)
        indices = torch.argmin(diff, dim=-1)
        x_quant = grid[indices] * scale
        return x_quant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def snap_weight(w):
    scale = w.abs().max().detach() + 1e-6
    return PrimalSTE.apply(w, scale)

class PrimalLayer(nn.Linear):
    def forward(self, input):
        w_quant = snap_weight(self.weight)
        return F.linear(input, w_quant, self.bias)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, max_len=512):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = PrimalLayer(n_embd, 3 * n_embd)
        self.c_proj = PrimalLayer(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc    = PrimalLayer(n_embd, 4 * n_embd)
        self.c_proj  = PrimalLayer(4 * n_embd, n_embd)
        self.act     = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PrimalBrain(nn.Module):
    def __init__(self, vocab_size=50257, n_layer=12, n_embd=384, n_head=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(512, n_embd)
        
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = PrimalLayer(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Primal-Expanse Brain on {device}...")
    
    # 12 Layers, 384 Dim => ~22M Params
    model = PrimalBrain(n_layer=12, n_embd=384, n_head=8).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Cosine Scheduler Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4) # Higher initial LR for Scheduler
    max_steps = 2000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)
    
    print("Loading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-111M")
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 8
    acc_steps = 4 # Accumulate 4 steps -> Effective Batch 32
    
    model.train()
    
    tokens = []
    step = 0
    running_loss = 0.0
    start_time = time.time()
    
    print("Starting Traning Loop (2000 Steps)...")
    for i, item in enumerate(dataset):
        text = item['text']
        enc = tokenizer.encode(text, truncation=True, max_length=512)
        if len(enc) < 32: continue 
        
        tokens.append(torch.tensor(enc))
        
        if len(tokens) == batch_size:
            max_len = max([len(t) for t in tokens])
            batch = torch.full((batch_size, max_len), tokenizer.eos_token_id, dtype=torch.long)
            for j, t in enumerate(tokens):
                batch[j, :len(t)] = t
                
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step() # Update LR
            optimizer.zero_grad()
            
            step += 1
            running_loss = 0.9 * running_loss + 0.1 * loss.item() if step > 1 else loss.item()
            
            if step % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | Avg: {running_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")
                
            tokens = []
            if step >= max_steps: break

    print("Training Complete.")
    torch.save(model.state_dict(), "primal_expanse.pt")

if __name__ == "__main__":
    train()
