import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from qwen25_prime_wrapper import PrimeQwen2CausalLM
from manifolds import generate_int16_prime_manifold
import os
import json

def run_test():
    print("[*] Starting fast Word Salad test...")
    
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

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    lut = generate_int16_prime_manifold(device=CONFIG['device'])
    model = PrimeQwen2CausalLM(CONFIG, lut=lut).to(CONFIG['device'])
    
    checkpoint = "qwen25_coder_prime_step_200.pt"
    if os.path.exists(checkpoint):
        print(f"[*] Loading checkpoint: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location=CONFIG['device']))
    
    model.eval()
    
    sample_prompt = "<|im_start|>user\nWrite a python function to print hello world\n<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(sample_prompt, return_tensors="pt").to(CONFIG['device'])
    input_ids = inputs["input_ids"]
    
    generated_ids = []
    temperature = 0.7
    top_p = 0.8
    repetition_penalty = 1.05
    
    print("[*] Sampling tokens...")
    with torch.no_grad():
        for i in range(30):
            logits, _ = model(input_ids)
            next_token_logits = logits[:, -1, :]
            
            # Penalties
            if repetition_penalty != 1.0 and len(generated_ids) > 0:
                for token_id in set(generated_ids):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty

            next_token_logits = next_token_logits / temperature
            
            # Top-P
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() in [tokenizer.eos_token_id, 151645]:
                break
                
    salad = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"\n[OUTPUT]:\n{salad}")

if __name__ == "__main__":
    run_test()
