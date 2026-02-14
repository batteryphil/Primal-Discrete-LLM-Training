import time
import os
import sys
import re

# Force UTF-8 for console output
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Older python versions
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

LOG_FILE = "training.log"

def format_line(line):
    """
    Parses and colorizes the log line for better visibility.
    Format: Step 95 | Loss: 5.0178 | TPS: 5460.41 | Flips: 0.0353% | P-Scale: 0.9997 | Anneal: 0.97 | VRAM: 6.07GB
    """
    line = line.strip()
    if not line: return None
    
    # Check if it's a standard metric line
    if "Step" in line and "Loss" in line:
        # High visibility colors (ANSI)
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        
        # Regex to pull metrics
        def get_val(key):
            match = re.search(fr"{key}:\s*([\d\.]+)", line)
            return match.group(1) if match else "?"

        step_match = re.search(r"Step (\d+)", line)
        step = step_match.group(1) if step_match else "?"
        
        loss = get_val("Loss")
        tps = get_val("TPS")
        flips = get_val("Flips")
        pscale = get_val("P-Scale")
        
        # Determine health colors
        loss_col = GREEN if float(loss) < 6.0 else YELLOW if float(loss) < 8.0 else RED
        flip_col = GREEN if float(flips) < 0.05 else YELLOW if float(flips) < 0.1 else RED
        
        return f"{BOLD}{CYAN}[Step {step}]{RESET} | {loss_col}Loss: {loss}{RESET} | {YELLOW}TPS: {tps}{RESET} | {flip_col}Flips: {flips}%{RESET} | P-Scale: {pscale}"
    
    # Pass through other logs (Avalanches, thermal resets, etc)
    if "[!]" in line:
        return f"\033[91m{line}\033[0m" # RED for alerts
    if "[*]" in line:
        return f"\033[94m{line}\033[0m" # BLUE for info
        
    return line

def monitor():
    print("="*60)
    print(f"[*] PRIMAL-DISCRETE TRAINING MONITOR")
    print(f"[*] Target: {os.path.abspath(LOG_FILE)}")
    print("="*60)
    
    # Wait for log file to appear
    while not os.path.exists(LOG_FILE):
        print(f"[!] Waiting for {LOG_FILE}...")
        time.sleep(1)

    print("[*] Monitoring from start of log...")
    
    try:
        while True:
            try:
                with open(LOG_FILE, "r", encoding='utf-8', errors='replace') as f:
                    # Start from beginning to catch everything
                    last_pos = 0
                    while True:
                        # Detect file truncation (training restarted)
                        try:
                            file_size = os.path.getsize(LOG_FILE)
                        except OSError:
                            time.sleep(1)
                            break  # File gone, reopen
                        
                        if file_size < last_pos:
                            print("\n\033[93m[*] Log file reset detected — training restarted.\033[0m")
                            break  # Reopen from start
                        
                        f.seek(last_pos)
                        line = f.readline()
                        if not line:
                            time.sleep(0.5)
                            continue
                        
                        last_pos = f.tell()
                        formatted = format_line(line)
                        if formatted:
                            print(formatted)
                            sys.stdout.flush()
            except (IOError, OSError):
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Monitor stopped.")

if __name__ == "__main__":
    monitor()
