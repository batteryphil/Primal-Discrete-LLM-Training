import os
import numpy as np
import torch
import librosa
import soundfile as sf
import io
from datasets import load_dataset
from g2p_en import G2p
from tqdm import tqdm
import json
import warnings
import nltk

try:
    nltk.data.find('averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# FastSpeech 2 / Ghost-TTS Config
CONFIG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "f_min": 0,
    "f_max": 8000,
    "n_mels": 80,
    "speaker_id": "p294", # British Female (Swapped for immediate start)
    "output_dir": "data/vctk_p225"
}

warnings.filterwarnings("ignore")

def extract_features(audio, sr):
    if sr != CONFIG["sample_rate"]:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=CONFIG["sample_rate"])
    
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=CONFIG["sample_rate"], n_fft=CONFIG["n_fft"], 
        hop_length=CONFIG["hop_length"], win_length=CONFIG["win_length"], 
        n_mels=CONFIG["n_mels"], fmin=CONFIG["f_min"], fmax=CONFIG["f_max"]
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80) / 80 # Normalize to [0, 1] approximately

    # F0 (Pitch)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), 
        sr=CONFIG["sample_rate"], hop_length=CONFIG["hop_length"]
    )
    f0[np.isnan(f0)] = 0

    # Energy
    energy = librosa.feature.rms(y=audio, frame_length=CONFIG["win_length"], hop_length=CONFIG["hop_length"])[0]

    return audio, mel.T, f0, energy

def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(CONFIG['output_dir'], "wavs"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG['output_dir'], "mels"), exist_ok=True)
    
    print(f"[*] Streaming VCTK dataset (Searching for Speaker: {CONFIG['speaker_id']})...")
    
    try:
        dataset = load_dataset("badayvedat/VCTK", split="train", streaming=True)
    except Exception as e:
        print(f"[!] Failed to load dataset: {e}")
        return

    metadata = []
    g2p = G2p()
    # Warmup G2P
    _ = g2p("warmup")
    
    count = 0
    max_items = 1000 # Target dataset size
    scanned = 0
    
    for item in tqdm(dataset):
        scanned += 1
        spk = item.get('speaker_id')
        if isinstance(spk, bytes):
            spk = spk.decode('utf-8')
            
        if spk != CONFIG['speaker_id']:
            if scanned % 100 == 0:
                # print(f"[*] Scanning... ({scanned} checked)", end='\r')
                pass
            continue
            
        try:
            text = item['txt']
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            
            # [FIX] 'flac' key contains decoded audio dict
            flac_data = item['flac']
            audio_data = np.array(flac_data['array'])
            sampling_rate = flac_data['sampling_rate']
            
            file_id = f"{CONFIG['speaker_id']}_{count:04d}"

            # 1. Phonemize
            phonemes = g2p(text)
            phoneme_str = " ".join([p for p in phonemes if p != ' '])

            # 2. Extract
            wav, mel, f0, energy = extract_features(audio_data, sampling_rate)

            # --- Z-NORMALIZATION (Phase 65) ---
            # Pitch (Log-scale, then Z-Norm non-zeros)
            f0_nonzero = f0[f0 > 0]
            if len(f0_nonzero) > 0:
                f0_log = np.log(f0_nonzero)
                f0_mean = np.mean(f0_log)
                f0_std = np.std(f0_log) + 1e-6
                f0_norm = np.zeros_like(f0)
                f0_norm[f0 > 0] = (np.log(f0[f0 > 0]) - f0_mean) / f0_std
            else:
                f0_norm = np.zeros_like(f0)

            # Energy (Log-scale, then Z-Norm)
            # Energy is usually > 0, but add eps
            energy_log = np.log(energy + 1e-6)
            energy_mean = np.mean(energy_log)
            energy_std = np.std(energy_log) + 1e-6
            energy_norm = (energy_log - energy_mean) / energy_std

            # 3. Save
            wav_path = os.path.join("wavs", f"{file_id}.wav")
            mel_path = os.path.join("mels", f"{file_id}.npy")
            pitch_path = os.path.join("pitch", f"{file_id}.npy")
            energy_path = os.path.join("energy", f"{file_id}.npy")
            
            sf.write(os.path.join(CONFIG['output_dir'], wav_path), wav, CONFIG["sample_rate"])
            np.save(os.path.join(CONFIG['output_dir'], mel_path), mel)
            np.save(os.path.join(CONFIG['output_dir'], pitch_path), f0_norm)
            np.save(os.path.join(CONFIG['output_dir'], energy_path), energy_norm)

            metadata.append({
                "id": file_id,
                "text": text,
                "phonemes": phoneme_str,
                "duration_frames": mel.shape[0],
                "f0_mean": float(np.mean(f0)), # Keep raw stats for reference
                "f0_std": float(np.std(f0)),
                "energy_mean": float(np.mean(energy)),
                "energy_std": float(np.std(energy))
            })

            count += 1
            if count % 10 == 0:
                with open(os.path.join(CONFIG['output_dir'], "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                print(f"[*] Progressive save: {count} items.")
                
            if count >= max_items:
                break
                
        except Exception as e:
            print(f"[!] Error processing item: {e}")
            continue

    if metadata:
        with open(os.path.join(CONFIG['output_dir'], "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    print(f"[*] Done. Processed {count} items for {CONFIG['speaker_id']}.")

if __name__ == "__main__":
    main()
