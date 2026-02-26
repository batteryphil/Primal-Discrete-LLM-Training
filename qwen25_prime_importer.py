import torch
from transformers import AutoModelForCausalLM
from manifolds import generate_int16_prime_manifold
from qwen25_prime_wrapper import PrimeQwen2CausalLM, normalize_weight_for_lut, fp16_weight_to_lut_index

# ==============================================================================
# QWEN2.5-CODER-1.5B PRIME IMPORTER
# ==============================================================================
# Transplanting FP16 knowledge into 16-bit Prime-Harmonic vote buffers.
# ==============================================================================

def transplant():
    print("[*] Loading Qwen2.5-Coder-1.5B-Instruct (CPU)...")
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    config = {
        'hidden_size': 1536,
        'num_hidden_layers': 28,
        'num_attention_heads': 12,
        'num_key_value_heads': 2,
        'intermediate_size': 8960,
        'vocab_size': 151936,
        'rms_norm_eps': 1e-6,
        'rope_theta': 1000000,
        'max_position_embeddings': 2048
    }

    print("[*] Generating 16-bit Prime-Harmonic Manifold...")
    lut = generate_int16_prime_manifold(device='cpu')

    print("[*] Initializing PRIME model shell...")
    prime = PrimeQwen2CausalLM(config, lut=lut)

    print("[*] Starting transplant...")
    
    # 1. Embeddings (untied)
    print("  [>] Tokens...")
    embed_w = qwen.model.embed_tokens.weight.data.float()
    norm_embed, _ = normalize_weight_for_lut(embed_w) # Embeddings don't use scale in TrueShadowless
    base, fine = fp16_weight_to_lut_index(norm_embed, lut)
    prime.embed_tokens.base_idx.copy_(base)
    prime.embed_tokens.fine_idx.copy_(fine)

    # 2. Layers
    attn_projs = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    mlp_projs = ['gate_proj', 'up_proj', 'down_proj']

    for i in range(28):
        print(f"  [>] Layer {i}/27...")
        qwen_layer = qwen.model.layers[i]
        prime_layer = prime.layers[i]

        # Attention
        for proj in attn_projs:
            q_w = getattr(qwen_layer.self_attn, proj).weight.data.float()
            ghost = getattr(prime_layer.self_attn, proj)
            norm_w, scale = normalize_weight_for_lut(q_w)
            base, fine = fp16_weight_to_lut_index(norm_w, lut)
            ghost.base_idx.copy_(base)
            ghost.fine_idx.copy_(fine)
            ghost.scale.data.copy_(scale)

        # MLP
        for proj in mlp_projs:
            m_w = getattr(qwen_layer.mlp, proj).weight.data.float()
            ghost = getattr(prime_layer.mlp, proj)
            norm_w, scale = normalize_weight_for_lut(m_w)
            base, fine = fp16_weight_to_lut_index(norm_w, lut)
            ghost.base_idx.copy_(base)
            ghost.fine_idx.copy_(fine)
            ghost.scale.data.copy_(scale)

        # Layernorms
        prime_layer.input_layernorm.weight.data.copy_(qwen_layer.input_layernorm.weight.data)
        prime_layer.post_attention_layernorm.weight.data.copy_(qwen_layer.post_attention_layernorm.weight.data)

    # 3. Final Head
    print("  [>] LM Head...")
    head_w = qwen.lm_head.weight.data.float()
    norm_head, scale_head = normalize_weight_for_lut(head_w)
    base, fine = fp16_weight_to_lut_index(norm_head, lut)
    prime.lm_head.base_idx.copy_(base)
    prime.lm_head.fine_idx.copy_(fine)
    prime.lm_head.scale.data.copy_(scale_head)

    # 4. Final Norm
    prime.norm.weight.data.copy_(qwen.model.norm.weight.data)

    print("[*] Saving PRIME checkpoint...")
    torch.save(prime.state_dict(), "qwen25_coder_prime_init.pt")
    print("[✓] Transplant Complete: qwen25_coder_prime_init.pt")

if __name__ == "__main__":
    transplant()
