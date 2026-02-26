import time
import os
import sys
import re

# Force UTF-8 for console output
import codecs
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

LOG_FILE = "training_qwen.log"

def format_line(line):
    """
    Accuracy Recovery Monitor v1.2
    Format: Step 1 | Loss: 2.1450 | Flips: 0 | TPS: 34.12 | Time: 7.72s
    OR Heartbeat: [10/32] L:2.1450 TPS:4.5
    """
    line = line.strip()
    if not line: return None
    
    # Handle Batch Heartbeats
    if "[" in line and "/" in line and "TPS:" in line:
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        RESET = "\033[0m"
        match = re.search(r"(\[\d+/\d+\])\s*L:([\d\.\-]+)\s*TPS:([\d\.\-]+)", line)
        if match:
            batch, loss, tps = match.groups()
            return f"  {CYAN}{batch}{RESET} > Batch Loss: {loss} | {YELLOW}Real-time TPS: {tps}{RESET}"

    # Check if it's a standard metric line
    if "Step" in line and "Loss" in line:
        # High visibility colors (ANSI)
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        
        # Regex to pull metrics
        def get_val(key):
            match = re.search(fr"{key}:\s*([naninf\d\.\-]+)", line, re.IGNORECASE)
            return match.group(1) if match else "?"

        def safe_float(val, default=0.0):
            try: return float(val)
            except: return default

        step_match = re.search(r"Step (\d+)", line)
        step_val = int(step_match.group(1)) if step_match else 0
        
        loss_str = get_val("Loss")
        flips_str = get_val("Flips")
        tps_str = get_val("TPS")
        
        loss = safe_float(loss_str)
        flips = safe_float(flips_str)
        
        # Determine Phase
        if step_val < 200:
            phase = f"{MAGENTA}STABILIZATION (Head/Embed Only){RESET}"
            peel = "Locked [0-27]"
        else:
            unlocked = min(((step_val - 200) // 50 + 1) * 2, 28)
            phase = f"{CYAN}LAYER PEEL (Discrete Healing){RESET}"
            peel = f"Unlocked [{28-unlocked}-27]"

        # Determine health colors
        loss_col = GREEN if loss < 4.0 else YELLOW if loss < 6.0 else RED
        
        return f"{BOLD}{CYAN}[Step {step_val}]{RESET} | {phase} | {loss_col}Loss: {loss_str}{RESET} | Flips: {flips_str} | TPS: {tps_str} | {BOLD}{peel}{RESET}"
    
    # Pass through Night Shift Alerts
    if "[NIGHT SHIFT]" in line:
        return f"\033[93m{line}\033[0m" # YELLOW
    if "[!]" in line:
        return f"\033[91m{line}\033[0m" # RED
    if "[*]" in line:
        return f"\033[94m{line}\033[0m" # BLUE
        
    return line

def monitor():
    print("="*80)
    print(f"{'ACCURACY RECOVERY MONITOR':^80}")
    print(f"{'Target: Qwen2.5-Coder-1.5B PRIME':^80}")
    print("="*80)
    
    # Wait for log file to appear
    while not os.path.exists(LOG_FILE):
        print(f"[!] Waiting for {LOG_FILE}...", end="\r")
        time.sleep(1)

    print(f"\033[K[*] Monitoring: {os.path.abspath(LOG_FILE)}")
    
    last_pos = 0
    try:
        while True:
            if not os.path.exists(LOG_FILE):
                time.sleep(1)
                continue
                
            file_size = os.path.getsize(LOG_FILE)
            if file_size < last_pos:
                print("\n\033[93m[*] Log reset detected.\033[0m")
                last_pos = 0

            with open(LOG_FILE, "r", encoding='utf-8', errors='replace') as f:
                f.seek(last_pos)
                lines = f.readlines()
                if lines:
                    for line in lines:
                        formatted = format_line(line)
                        if formatted:
                            print(formatted)
                    last_pos = f.tell()
                else:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[*] Monitor stopped.")

if __name__ == "__main__":
    monitor()
