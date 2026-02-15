import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import random
import time
from tqdm import tqdm

from ghost_tts import GhostTTS, CONFIG as TTS_CONFIG

# --- CONFIGURATION ---
TRAIN_CONFIG = {
    "batch_size": 4, # Reduced for stability
    "epochs": 100,
    "lr": 1e-4, # [HOTFIX] Reduced from 1e-3 to prevent rapid overfitting
    "warmup_steps": 1000,
    "log_interval": 10,
    "save_interval": 500,
    "checkpoint_dir": "checkpoints/ghost_tts",
    "data_dir": "data/vctk_p225", # Hardcoded to match pipeline
    "device": TTS_CONFIG["device"],
    "grad_accum_steps": 4, # Virtual Batch Size = 4 * 4 = 16
    "num_workers": 4,
    "pin_memory": True,
    "val_split": 0.1,
    "lambda_pitch": 1.0, # [HOTFIX] Normalized from 5.0
    "lambda_energy": 1.0 # [HOTFIX] Normalized from 5.0
}

# --- DATASET ---
class GhostDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.data_dir = os.path.dirname(metadata_path)
        
        # Build phoneme vocab (simple char-level for now or use g2p set)
        # We need a consistent mapping.
        # Let's collect all unique phonemes.
        phonemes = set()
        for item in self.metadata:
             for p in item['phonemes'].split():
                 phonemes.add(p)
        
        self.phoneme_to_id = {p: i+1 for i, p in enumerate(sorted(list(phonemes)))} # 0 is pad
        self.phoneme_to_id["<pad>"] = 0
        print(f"[*] Vocab Size: {len(self.phoneme_to_id)}")
        
        # Save Vocab for Inference
        vocab_path = os.path.join(self.data_dir, "vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(self.phoneme_to_id, f, indent=2)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load Mel
        mel_path = os.path.join(self.data_dir, "mels", f"{item['id']}.npy")
        mel = np.load(mel_path) # [T, 80]
        mel = torch.from_numpy(mel) # [T, 80]
        
        # Phonemes
        phons = [self.phoneme_to_id.get(p, 0) for p in item['phonemes'].split()]
        phons = torch.tensor(phons, dtype=torch.long)
        
        # Derived targets (For now, we don't have aligned durations!)
        # FastSpeech 2 requires alignment (MFA).
        # Since we are doing a "speed run" without MFA, we use a hack:
        # Uniform alignment or Attention distilling.
        # BUT, to keep it simple and runnable without MFA:
        # We will use Length Regulator with "predicted" duration during training? No.
        # We need ground truth duration.
        # Fallback: Use a simple autoregressive duration proxy or just learn to expand equally?
        # Actually, for this specific "11GB Challenge" task, maybe we should use
        # explicit alignment if possible.
        # But setting up MFA is hard.
        # Alternative: Use an existing alignment tool (Rez? Montreal Forced Aligner?)
        # Or... use a Tacotron-style attention to get alignment?
        # Ghost-TTS is Non-Autoregressive.
        # A common hack: Divide total Mel frames by phonemes to get average duration.
        # It's bad, but it runs.
        # Let's use "Uniform Alignment" for the MVP.
        # duration = mel_len / phoneme_len
        
        mel_len = mel.shape[0]
        phon_len = len(phons)
        duration = torch.zeros(phon_len, dtype=torch.long)
        # Distribute frames
        # e.g. 100 frames, 10 phonemes -> 10 each
        base = mel_len // phon_len
        rem = mel_len % phon_len
        duration[:] = base
        duration[:rem] += 1
        
        # Pitch/Energy (Load Pre-Computed Z-Normed Latents)
        pitch_path = os.path.join(self.data_dir, "pitch", f"{item['id']}.npy")
        energy_path = os.path.join(self.data_dir, "energy", f"{item['id']}.npy")
        
        if os.path.exists(pitch_path):
            pitch = torch.from_numpy(np.load(pitch_path))
            energy = torch.from_numpy(np.load(energy_path))
        else:
            # Fallback if user hasn't re-run pipeline yet
            # print(f"[!] Warning: Missing pitch/energy for {item['id']}")
            pitch = torch.zeros(mel_len)
            energy = torch.zeros(mel_len)
        
        return {
            "id": item['id'],
            "text": phons,
            "mel": mel,
            "duration": duration,
            "pitch": pitch,
            "energy": energy
        }

def collate_fn(batch):
    # dynamic padding
    ids = [x['id'] for x in batch]
    text_lens = [len(x['text']) for x in batch]
    mel_lens = [len(x['mel']) for x in batch]
    
    max_text = max(text_lens)
    max_mel = max(mel_lens)
    
    text_padded = torch.zeros(len(batch), max_text, dtype=torch.long)
    mel_padded = torch.zeros(len(batch), max_mel, 80, dtype=torch.float)
    duration_padded = torch.zeros(len(batch), max_text, dtype=torch.long)
    pitch_padded = torch.zeros(len(batch), max_mel, dtype=torch.float)
    energy_padded = torch.zeros(len(batch), max_mel, dtype=torch.float)
    
    for i, x in enumerate(batch):
        text_padded[i, :text_lens[i]] = x['text']
        mel_padded[i, :mel_lens[i]] = x['mel']
        duration_padded[i, :text_lens[i]] = x['duration']
        pitch_padded[i, :mel_lens[i]] = x['pitch']
        energy_padded[i, :mel_lens[i]] = x['energy']
        
    return {
        "text": text_padded,
        "mel": mel_padded,
        "duration": duration_padded,
        "pitch": pitch_padded,
        "energy": energy_padded
    }

# --- LOSS FUNCTIONS ---
class GhostLoss(nn.Module):
    def __init__(self, lambda_pitch=5.0, lambda_energy=5.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.lambda_pitch = lambda_pitch
        self.lambda_energy = lambda_energy

    def forward(self, model_output, targets):
        (mel_pred, mel_post_pred, log_dur_pred, pitch_pred, energy_pred, src_mask) = model_output
        (mel_target, duration_target, pitch_target, energy_target) = targets
        
        # Mel Loss
        loss_mel = self.l1(mel_pred, mel_target)
        loss_mel_post = self.l1(mel_post_pred, mel_target)
        
        # Duration Loss (Log domain)
        log_duration_target = torch.log(duration_target.float() + 1)
        loss_duration = self.mse(log_dur_pred, log_duration_target)
        
        # Pitch/Energy Loss
        loss_pitch = self.mse(pitch_pred, pitch_target)
        loss_energy = self.mse(energy_pred, energy_target)
        
        # [PHASE 66] Variance for Watchdog
        with torch.no_grad():
            pitch_std = pitch_pred.std().item()
            energy_std = energy_pred.std().item()
        
        total_loss = loss_mel + loss_mel_post + loss_duration + (self.lambda_pitch * loss_pitch) + (self.lambda_energy * loss_energy)
        
        return total_loss, {
            "mel": loss_mel.item(),
            "dur": loss_duration.item(),
            "pit": loss_pitch.item(),
            "ene": loss_energy.item(),
            "p_std": pitch_std,
            "e_std": energy_std
        }

# --- THE SENTINEL PROTOCOL ---
class SentinelProtocol:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Watchdog state
        self.flatline_counter = 0
        self.last_std_threshold = 0.005
        
        # Stability state
        self.grad_norms = []
        
        # Early Stopping state
        self.train_loss_history = []
        self.val_loss_history = []
        self.patience = 5
        self.counter = 0
        self.best_val_loss = float('inf')
        
    def step_batch(self, loss, scaler):
        """Called during grad accumulation loop."""
        # Gradient Stability
        # Check Total Norm before clipping
        scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000.0) # Dummy high clip to get norm
        
        # Hard Clip if exploding
        if total_norm > 10.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
        # Reduce LR if vanishing
        if total_norm < 1e-6:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.8
                
        return total_norm.item()

    def step_epoch(self, epoch, train_mel, val_mel, avg_p_std, avg_e_std):
        """Called at end of epoch."""
        # 1. Prosody Flatline Watchdog
        if avg_p_std < self.last_std_threshold or avg_e_std < self.last_std_threshold:
            self.flatline_counter += 1
            if self.flatline_counter >= 3:
                print(f"\n[!] SENTINEL: Prosody Flatline Detected (P:{avg_p_std:.4f}, E:{avg_e_std:.4f}). Triggering resurrection...")
                # Re-init predictors
                self.model.pitch_predictor.re_init()
                self.model.energy_predictor.re_init()
                # Double weights
                self.criterion.lambda_pitch *= 2.0
                self.criterion.lambda_energy *= 2.0
                print(f"[*] New Weights: Pitch={self.criterion.lambda_pitch}, Energy={self.criterion.lambda_energy}")
                self.flatline_counter = 0
        else:
            self.flatline_counter = 0
            
        # 2. Early Stopping Guard
        self.train_loss_history.append(train_mel)
        self.val_loss_history.append(val_mel)
        
        if val_mel < self.best_val_loss:
            self.best_val_loss = val_mel
            self.counter = 0
            # Save "Best" so far
            torch.save(self.model.state_dict(), os.path.join(TRAIN_CONFIG["checkpoint_dir"], "best_sentinel.pt"))
        else:
            # Check if training is still dropping but val is rising
            if train_mel < self.train_loss_history[-2] if len(self.train_loss_history) > 1 else train_mel:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"\n[!] SENTINEL: Overfitting Detected. Triggering Early Stop.")
                    return True # Stop
        return False

# --- TRAINING LOOP ---
def train():
    os.makedirs(TRAIN_CONFIG["checkpoint_dir"], exist_ok=True)
    
    # Check data
    meta_path = os.path.join(TRAIN_CONFIG["data_dir"], "metadata.json")
    if not os.path.exists(meta_path):
        print(f"[!] Metadata not found at {meta_path}. Run tts_data_pipeline.py first.")
        return

    # Dataset Split
    full_dataset = GhostDataset(meta_path)
    val_size = int(len(full_dataset) * TRAIN_CONFIG["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=TRAIN_CONFIG["pin_memory"]
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        collate_fn=collate_fn,
        num_workers=TRAIN_CONFIG["num_workers"]
    )
    
    # Model
    model = GhostTTS(vocab_size=len(full_dataset.phoneme_to_id)+1).to(TRAIN_CONFIG["device"])
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"], betas=(0.9, 0.98), eps=1e-9)
    scaler = torch.cuda.amp.GradScaler() # AMP
    criterion = GhostLoss(lambda_pitch=TRAIN_CONFIG["lambda_pitch"], lambda_energy=TRAIN_CONFIG["lambda_energy"])
    sentinel = SentinelProtocol(model, optimizer, criterion)
    
    print(f"[*] Model Size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    
    step = 0
    model.train()
    
    for epoch in range(TRAIN_CONFIG["epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            text = batch['text'].to(TRAIN_CONFIG["device"])
            mel_target = batch['mel'].to(TRAIN_CONFIG["device"])
            duration_target = batch['duration'].to(TRAIN_CONFIG["device"])
            pitch_target = batch['pitch'].to(TRAIN_CONFIG["device"])
            energy_target = batch['energy'].to(TRAIN_CONFIG["device"])
            # Forward with AMP
            with torch.cuda.amp.autocast():
                output = model(text, mel_target, duration_target, pitch_target, energy_target)
                
                # Loss
                loss, metrics = criterion(output, (mel_target, duration_target, pitch_target, energy_target))
                loss = loss / TRAIN_CONFIG["grad_accum_steps"] # Normalize
            
            pbar.set_postfix({
                "loss": f"{loss.item() * TRAIN_CONFIG['grad_accum_steps']:.4f}",
                "mel": f"{metrics['mel']:.4f}",
                "dur": f"{metrics['dur']:.4f}",
                "pit": f"{metrics['pit']:.4f}",
                "ene": f"{metrics['ene']:.4f}" 
            })
            
            # Backward with Scaler
            scaler.scale(loss).backward()
            
            if (step + 1) % TRAIN_CONFIG["grad_accum_steps"] == 0:
                # [PHASE 66] Sentinel Batch Step (Stability Tracker)
                grad_norm = sentinel.step_batch(loss, scaler)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Ghost Quantization Updates
                # Annealing? Start high sensitivity, lower it? Or vice versa?
                # GhostLinear logic: sensitivity is fixed, but we can anneal effect.
                # For now, dynamic update:
                if step % 5 == 0:
                    flips = model.apply_ghost_votes(adaptive_prob=0.01)

            step += 1
            
        # --- VALIDATION PASS ---
        model.eval()
        val_losses = []
        val_p_stds = []
        val_e_stds = []
        with torch.no_grad():
            for v_batch in val_dataloader:
                v_text = v_batch['text'].to(TRAIN_CONFIG["device"])
                v_mel = v_batch['mel'].to(TRAIN_CONFIG["device"])
                v_dur = v_batch['duration'].to(TRAIN_CONFIG["device"])
                v_pit = v_batch['pitch'].to(TRAIN_CONFIG["device"])
                v_ene = v_batch['energy'].to(TRAIN_CONFIG["device"])
                
                v_out = model(v_text, v_mel, v_dur, v_pit, v_ene)
                v_loss, v_metrics = criterion(v_out, (v_mel, v_dur, v_pit, v_ene))
                val_losses.append(v_metrics["mel"])
                val_p_stds.append(v_metrics["p_std"])
                val_e_stds.append(v_metrics["e_std"])
        
        avg_val_mel = float(np.mean(val_losses)) if val_losses else 0.0
        avg_train_mel = metrics["mel"]
        avg_p_std = float(np.mean(val_p_stds)) if val_p_stds else 0.0
        avg_e_std = float(np.mean(val_e_stds)) if val_e_stds else 0.0
        
        print(f"[*] Validation - Mel: {avg_val_mel:.4f}, P_Std: {avg_p_std:.4f}, E_Std: {avg_e_std:.4f}")
        
        # [PHASE 66] Sentinel Epoch Step (Watchdog & ES)
        if sentinel.step_epoch(epoch, avg_train_mel, avg_val_mel, avg_p_std, avg_e_std):
            break
            
        model.train()

        # Non-blocking Checkpoint at end of Epoch
        if (epoch + 1) % 10 == 0: # Save every 10 epochs
             torch.save(model.state_dict(), os.path.join(TRAIN_CONFIG["checkpoint_dir"], f"checkpoint_{epoch+1}.pt"))
             # print(f"[*] Saved checkpoint at epoch {epoch+1}")

    print("[*] Training Complete.")
    # Export best weights from Sentinel
    best_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "best_sentinel.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print("[*] Exported best validation weights.")
        
    torch.save(model.state_dict(), os.path.join(TRAIN_CONFIG["checkpoint_dir"], "final_model.pt"))

if __name__ == "__main__":
    train()
