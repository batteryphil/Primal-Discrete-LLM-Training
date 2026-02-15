import os
import subprocess
from tqdm import tqdm

TEST_SENTENCES = [
    "Project Trinity is alive. The Sentinel Protocol has successfully guarded the training session.",
    "The 1.58-bit engine is running with high fidelity prosody.",
    "This is a test of the continuous projection architecture.",
    "Ghost TTS is now synthesized using a neural vocoder."
]

CHECKPOINT = "checkpoints/ghost_tts/best_sentinel.pt"
OUTPUT_DIR = "tests/voice_samples"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[*] Starting Automated Voice Test...")
    print(f"[*] Checkpoint: {CHECKPOINT}")
    
    for i, text in enumerate(tqdm(TEST_SENTENCES)):
        output_file = os.path.join(OUTPUT_DIR, f"sample_{i+1}.wav")
        cmd = [
            "python", "tts_inference.py",
            "--text", text,
            "--checkpoint", CHECKPOINT,
            "--output", output_file
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[!] Error generating sample {i+1}:")
            print(e.stderr)
            continue
            
    print(f"\n[+] Automation Complete. Samples saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
