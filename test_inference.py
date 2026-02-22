import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import manifolds
import sys
import os
from ghost_core import PrimalTRMCore, GhostLinear, LogitTamer
from primal_train_modular import GhostGPT

# Model Configuration (Matching 0.1B Project Real)
CONFIG = {
    "dim": 768,
    "n_layers": 6,
    "n_heads": 6,
    "vocab_size": 50257,
    "seq_len": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "iterations": 8
}

def load_model(checkpoint_path):
    print(f"[*] Loading Antigravity 0.1B Model from {checkpoint_path}...")
    
    # Load Manifold (Linear 8-bit for Project Real)
    lut = manifolds.generate_linear_manifold(device=CONFIG['device'])
    
    # Initialize Model
    model = GhostGPT(CONFIG).to(CONFIG['device'])
    
    # Load Weights
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG['device']), strict=False)
            model.eval()
            print("[*] Weights loaded successfully.")
        except Exception as e:
            print(f"[!] Error loading weights: {e}")
            sys.exit(1)
    else:
        print(f"[!] Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
        
    return model, lut

def generate_response(model, lut, prompt, max_tokens=128):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(CONFIG['device'])
    
    print(f"[*] Generating response for: '{prompt}'...")
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=max_tokens)
        response = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    
    return response

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_inference.py \"Your question here\"")
        sys.exit(1)
        
    prompt = sys.argv[1]
    checkpoint = "project_real_live.pt"
    
    model, lut = load_model(checkpoint)
    result = generate_response(model, lut, prompt)
    
    print("\n" + "="*50)
    print("MODEL RESPONSE:")
    print("="*50)
    print(result)
    print("="*50)
