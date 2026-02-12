import ctypes
import torch
import numpy as np
import os
from transformers import AutoTokenizer
from primal_train_v2 import PrimalBrain

# 0. Path Config
DLL_PATH = "./src/engine/bin/primal_engine.dll"
MODEL_PATH = b"src/engine/model.primal"
CHECKPOINT_PATH = "primal_expanse.pt"

if not os.path.exists(DLL_PATH):
    print(f"Error: DLL not found at {DLL_PATH}")
    exit(1)

# 1. Setup DLL
print(f"Loading Engine from {DLL_PATH}...")
lib = ctypes.CDLL(DLL_PATH)

lib.create_engine.restype = ctypes.c_void_p
lib.create_engine.argtypes = [ctypes.c_char_p, ctypes.c_bool]

lib.run_inference.restype = None
lib.run_inference.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

lib.destroy_engine.restype = None
lib.destroy_engine.argtypes = [ctypes.c_void_p]

# 2. Load Python Side Components (Tokenizer + Embeddings)
print("Loading PrimalBrain Embeddings...")
try:
    # Load the same usage architecture
    tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-111M")
    
    # Load the trained PyTorch model to get embeddings
    py_model = PrimalBrain()
    py_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    py_model.eval()
    
    token_embedding = py_model.token_embedding
    pos_embedding = py_model.position_embedding
    
except Exception as e:
    print(f"Failed to load PrimalBrain components: {e}")
    exit(1)

# 3. Initialize Engine (C++ Side)
print("Initializing Primal Engine (Layers)...")
engine = lib.create_engine(MODEL_PATH, True) 

def chat(prompt):
    print(f"\nPrompt: {prompt}")
    tokens = tokenizer.encode(prompt)
    if not tokens:
        print("Error: Empty tokens")
        return

    # Prepare Input Vector (Embedding + Position)
    # This architecture requires [1, 256] input
    seq_len = len(tokens)
    last_token_id = tokens[-1]
    last_pos_id = seq_len - 1
    
    with torch.no_grad():
        t_emb = token_embedding(torch.tensor([last_token_id]))
        p_emb = pos_embedding(torch.tensor([last_pos_id]))
        # Sum them (Input to first transformer block)
        input_vec = (t_emb + p_emb).view(-1).numpy().astype(np.float32)
    
    # Prepare output buffer (Vocab Size = 50257)
    vocab_size = 50257
    output_buffer = (ctypes.c_float * vocab_size)()
    
    print(f"Running Inference (InSize={len(input_vec)})...")
    lib.run_inference(engine, input_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output_buffer, len(input_vec))
    
    # Logic: Top-K Sampling
    logits = np.ctypeslib.as_array(output_buffer)
    
    # 1. Apply Temperature
    temperature = 0.8
    logits /= temperature
    
    # 2. Top-K Filter
    k = 50
    # Find the k-th largest value
    ind = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[ind]
    top_k_indices = ind
    
    # Softmax
    exp_logits = np.exp(top_k_logits - np.max(top_k_logits)) # Stability
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample
    next_token_idx = np.random.choice(len(probs), p=probs)
    next_token = top_k_indices[next_token_idx]
    
    decoded = tokenizer.decode([next_token])
    print(f"Sampled Token ID: {next_token} (from Top-{k})")
    print(f"Predicted Next Token: '{decoded}'")

if __name__ == "__main__":
    try:
        chat("Once upon a time")
    finally:
        if 'engine' in locals():
            lib.destroy_engine(engine)
