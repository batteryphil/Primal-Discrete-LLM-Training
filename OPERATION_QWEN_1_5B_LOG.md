# Operation: Qwen2.5-Coder-1.5B PRIME Conversion

## Objective

Convert `Qwen/Qwen2.5-Coder-1.5B-Instruct` to the PRIME discrete format and train it on a GTX 1080 Ti (11GB VRAM).

## Checkpoint Strategy

Before any major execution or state change, we will backup the working script versions or verify model weights can be re-generated deterministically. If a training run crashes, we will have `micro_save_interval` checkpoints to fall back to.

---

## Log

### Step 1: Infrastructure Review & Setup

- **Status:** Complete.
- **Action:** Reviewed `qwen25_prime_wrapper.py`, `qwen25_prime_importer.py`, `qwen25_prime_train.py`, and `verify_qwen25_prime.py`.
- **Discovery (Failure Averted):** Discovered that the existing 7.1GB `qwen25_coder_prime_init.pt` file and all scripts were erroneously hardcoded to use `generate_int16_linear_manifold`. This would have trained the model on a standard 16-bit linear grid instead of the mathematically crucial Prime-Harmonic grid central to Project PRIMAL.
- **Correction:**
  1. Backed up the original incorrectly-mapped weights as `qwen25_coder_prime_init.pt.linear_backup`.
  2. Modified all three core scripts (`importer`, `train`, `verify`) to explicitly import and use `generate_int16_prime_manifold`.
  3. Re-ran `python verify_qwen25_prime.py` on the old Linear weights just to verify the structural integrity of the wrapper. (Success: Forward pass worked properly, confirming the architecture).

### Step 2: The Gentrification (Importing to Prime Grid)

- **Status:** In Progress.
- **Action:** Currently running `python qwen25_prime_importer.py` to pull down the FP16 Qwen2.5-Coder weights and snap them deterministically to the 16-bit Prime-Harmonic Grid. This will reconstruct the `qwen25_coder_prime_init.pt` payload.
