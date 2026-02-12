import time
import re
import os
import subprocess
import sys

LOG_FILE = "training.log"
TARGET_STEP = 2500

def monitor():
    print(f"[*] Monitoring {LOG_FILE} for Step {TARGET_STEP}...")
    print("[!] Benchmark will run on CPU to avoid interrupting active training (VRAM Protection).")
    
    if not os.path.exists(LOG_FILE):
        print(f"[!] {LOG_FILE} not found. Waiting...")
        while not os.path.exists(LOG_FILE):
            time.sleep(1)

    # Open file and go to end? No, we need to read new lines.
    # But if we restart, we might miss lines.
    # Let's just tail from end.
    
    f = open(LOG_FILE, "r", encoding='utf-8', errors='replace')
    f.seek(0, os.SEEK_END)
    
    triggered = False
    
    while True:
        line = f.readline()
        if not line:
            time.sleep(1)
            continue
            
        print(line, end="")
        
        # Parse Step
        match = re.search(r"Step (\d+)", line)
        if match:
            step = int(match.group(1))
            if step >= TARGET_STEP and not triggered:
                triggered = True
                print(f"\n\n[*] TRIANGULATION: Step {step} reached! Deploying Perplexity Benchmark...")
                
                # Check point might need a moment to save if it saves every 100 steps
                # If Step 2500 is a checkpoint step (2500 % 100 == 0), we should wait for save confirmation
                # The training loop prints "[*] Saved Live Checkpoint"
                # We can wait for that string if we want, or just wait 10 seconds.
                print("[*] Waiting 10s for checkpoint consistency...", flush=True)
                time.sleep(10)
                
                # Run Benchmark on CPU
                # We set CUDA_VISIBLE_DEVICES empty to force CPU
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""
                
                try:
                    subprocess.run([sys.executable, "primal_val_perplexity.py"], env=env, check=False)
                except Exception as e:
                    print(f"[!] Benchmark Failed: {e}")
                
                print("[*] Benchmark Complete. Resuming Monitoring...")
                # We don't exit, we keep monitoring?
                # The user might want to see subsequent logs.
                
if __name__ == "__main__":
    monitor()
