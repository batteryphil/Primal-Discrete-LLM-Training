import torch
import numpy as np
import os
import json
import soundfile as sf
from g2p_en import G2p
import argparse
import sys
import librosa
from speechbrain.inference.vocoders import HIFIGAN
import speechbrain.utils.fetching as fetching

# [WINDOWS FIX] Force SpeechBrain to copy instead of symlink (avoids WinError 1314)
# Sledgehammer: Monkeypatch the low-level link_with_strategy function
if not hasattr(fetching, '_original_link_with_strategy'):
    fetching._original_link_with_strategy = fetching.link_with_strategy
    def forced_link(src, dst, strategy=None):
        # Ignore requested strategy, always use COPY on Windows
        return fetching._original_link_with_strategy(src, dst, fetching.LocalStrategy.COPY)
    fetching.link_with_strategy = forced_link

# Add local path for importing ghost_tts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ghost_tts import GhostTTS, CONFIG as TTS_CONFIG

def load_hifigan(device="cuda"):
    print("[*] Initializing HiFi-GAN (SpeechBrain)...")
    # Note: 'speechbrain/tts-hifigan-ljspeech' is 22050Hz.
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech", 
        savedir="tmp_models", 
        run_opts={"device": device}
    )
    return hifi_gan

def synthesize(text, checkpoint_path, output_path, vocab_path="data/vctk_p225/vocab.json", device="cuda"):
    
    # Check paths
    if not os.path.exists(vocab_path):
        print(f"[!] Vocab not found at {vocab_path}. Using default p294 path...")
        vocab_path = "data/vctk_p294/vocab.json"
        
    print(f"[*] Loading vocab from {vocab_path}...")
    try:
        with open(vocab_path, 'r') as f:
            phoneme_to_id = json.load(f)
    except FileNotFoundError:
        print("[!] Vocab file completely missing. Cannot proceed.")
        return

    # Load Ghost-TTS Model
    print(f"[*] Loading Ghost-TTS from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    chk_vocab_size = state_dict['embedding.weight'].shape[0]
    model = GhostTTS(vocab_size=chk_vocab_size).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Load Vocoder
    hifi_gan = load_hifigan(device)
    
    # G2P
    g2p = G2p()
    print(f"[*] Phonemizing: '{text}'")
    phonemes = g2p(text)
    print(f"    Phonemes: {phonemes}")
    
    phons = [phoneme_to_id.get(p, 0) for p in phonemes if p != ' ']
    phons_tensor = torch.tensor(phons, dtype=torch.long).unsqueeze(0).to(device)
    
    print("[*] Running Ghost-TTS Inference...")
    with torch.no_grad():
        (mel_pred, mel_post_pred, log_dur, pitch, energy, src_mask) = model(phons_tensor)
    
    # Debug Mel Stats
    print(f"    Mel Pred Stats - Min: {mel_pred.min().item():.4f}, Max: {mel_pred.max().item():.4f}, Mean: {mel_pred.mean().item():.4f}")
    
    # Post-Processing
    # Ghost-TTS Training Logic: mel_norm = (mel_db + 80) / 80
    # Inverse to dB:
    mel_db = (mel_post_pred * 80) - 80
    
    # HiFi-GAN expects Log-Mel: ln(Power)
    # dB = 10 * log10(Power)  => Power = 10^(dB/10)
    # ln(Power) = ln(10^(dB/10)) = (dB/10) * ln(10)
    log_mel = (mel_db / 10.0) * np.log(10.0)
    
    print(f"    Log-Mel Stats - Min: {log_mel.min().item():.4f}, Max: {log_mel.max().item():.4f}, Mean: {log_mel.mean().item():.4f}")

    # HiFi-GAN expects [Batch, Time, Channels] or [Batch, Channels, Time]?
    # SpeechBrain HIFIGAN.decode_batch expects [Batch, Time, Channels]
    start_mel = log_mel.transpose(1, 2) # [1, T, 80]
    
    print("[*] Synthesizing Audio (HiFi-GAN)...")
    with torch.no_grad():
        wav = hifi_gan.decode_batch(start_mel)
        
    wav = wav.squeeze().cpu().numpy()

    # Normalize Output
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav)) * 0.95
    
    sf.write(output_path, wav, 22050)
    print(f"[*] Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="The ghost in the machine is singing.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ghost_tts/checkpoint_3500.pt")
    parser.add_argument("--output", type=str, default="output.wav")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-find selection logic...
    if "checkpoint" not in args.checkpoint or not os.path.exists(args.checkpoint):
         if os.path.exists("checkpoints/ghost_tts"):
            ckpts = [f for f in os.listdir("checkpoints/ghost_tts") if f.endswith(".pt") and "final" not in f]
            if ckpts:
                def get_step(name):
                    try: return int(name.split("_")[1].split(".")[0])
                    except: return 0
                latest = max(ckpts, key=get_step)
                args.checkpoint = os.path.join("checkpoints/ghost_tts", latest)
                print(f"[*] Auto-selected latest checkpoint: {args.checkpoint}")
    
    synthesize(args.text, args.checkpoint, args.output, device=device)
