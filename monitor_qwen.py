import time
import os
import sys
import json
import re

# Force UTF-8 for console output
import codecs
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

LOG_FILE = "training_qwen.log"
STATS_FILE = "stats_coder.json"

# ANSI Colors for a premium feel
CYAN = "\033[36m"
BRIGHT_CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_SCREEN = "\033[2J\033[H"

def clear_screen():
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.flush()

def get_layer_peel_status(step):
    if step < 200:
        return f"{MAGENTA}PHASE 1: STABILIZATION{RESET} | Layers: {RED}Locked [0-27]{RESET} | Status: {YELLOW}Warming Embed/Head{RESET}"
    else:
        unlocked = min(((step - 200) // 50 + 1) * 2, 28)
        locked_remaining = 28 - unlocked
        if locked_remaining > 0:
            return f"{BRIGHT_CYAN}PHASE 2: LAYER PEEL{RESET} | Layers: {GREEN}Unlocked [{locked_remaining}-27]{RESET} ({unlocked}/28) | Status: {CYAN}Cascade Active{RESET}"
        else:
            return f"{GREEN}PHASE 3: FULL CRYSTALLIZATION{RESET} | Layers: {GREEN}ALL UNLOCKED (28/28){RESET} | Status: {GREEN}Complete{RESET}"

def render_dashboard(stats, latest_heartbeat, alerts):
    clear_screen()
    print(f"{BOLD}{BRIGHT_CYAN}{'='*80}{RESET}")
    print(f"{BOLD}{BRIGHT_CYAN}{'🚀 QWEN2.5-CODER-1.5B PRIME RECOVERY MONITOR':^80}{RESET}")
    print(f"{BOLD}{BRIGHT_CYAN}{'='*80}{RESET}")
    
    if not stats:
        print(f"\n{YELLOW}  Waiting for training to commence... (No stats found yet){RESET}\n")
    else:
        latest = stats[-1]
        step = latest.get("step", 0)
        loss = latest.get("loss", 0.0)
        tps = latest.get("tps", 0.0)
        flips = latest.get("flips", 0.0)
        lr = latest.get("lr", 0.0)

        # Loss health color
        loss_col = GREEN if loss < 4.0 else YELLOW if loss < 6.0 else RED
        
        print(f"\n  {BOLD}CURRENT STATE: {get_layer_peel_status(step)}{RESET}\n")
        
        print(f"  {BOLD}Step:{RESET} {BRIGHT_CYAN}{step:<8}{RESET} | "
              f"{BOLD}Loss:{RESET} {loss_col}{loss:<8.4f}{RESET} | "
              f"{BOLD}Flips:{RESET} {YELLOW}{flips:<8.4f}M{RESET} | "
              f"{BOLD}TPS:{RESET} {GREEN}{tps:<8.2f}{RESET} | "
              f"{BOLD}LR:{RESET} {MAGENTA}{lr}{RESET}")
    
    print(f"{DIM}{'-'*80}{RESET}")
    
    if latest_heartbeat:
        print(f"  {BOLD}{YELLOW}⚡ LIVE PULSE:{RESET} {latest_heartbeat}")
    else:
        print(f"  {BOLD}{DIM}⚡ LIVE PULSE: Listening for batch updates...{RESET}")
        
    print(f"{DIM}{'-'*80}{RESET}")
    
    if alerts:
        print(f"  {BOLD}{RED}⚠️ RECENT SYSTEM ALERTS:{RESET}")
        for alert in alerts[-5:]:
            print(f"    {alert}")
    else:
        print(f"  {BOLD}{GREEN}✓ System Nominal{RESET}")
        
    print(f"\n{DIM}Press Ctrl+C to exit monitor.{RESET}")

def run_monitor():
    last_pos = 0
    latest_heartbeat = ""
    alerts = []
    
    # Initialize UI
    render_dashboard([], "", [])
    
    try:
        while True:
            # 1. Load Stats
            stats = []
            if os.path.exists(STATS_FILE):
                try:
                    with open(STATS_FILE, "r") as f:
                        stats = json.load(f)
                except:
                    pass
            
            # 2. Tail Log File for Heartbeats and Alerts
            if os.path.exists(LOG_FILE):
                file_size = os.path.getsize(LOG_FILE)
                if file_size < last_pos:
                    alerts.append(f"Log reset detected at {time.strftime('%H:%M:%S')}")
                    last_pos = 0

                with open(LOG_FILE, "r", encoding='utf-8', errors='replace') as f:
                    f.seek(last_pos)
                    new_data = f.read()
                    if new_data:
                        last_pos = f.tell()
                        
                        # Extract heartbeats (e.g. [10/32] L:2.45 TPS:3.2)
                        # Because they are printed with end="" we regex search the raw string
                        heartbeats = re.findall(r"(\[\d+/\d+\]\s*L:[\d\.\-]+\s*TPS:[\d\.\-]+)", new_data)
                        if heartbeats:
                            latest_heartbeat = heartbeats[-1]
                            
                        # Extract alerts line by line
                        for line in new_data.split('\n'):
                            if "[NIGHT SHIFT]" in line or "[!]" in line or "[SYNCHRONIZE]" in line or "[*]" in line:
                                # Keep it clean
                                clean_line = line.strip().replace("\n", "")
                                if clean_line and clean_line not in alerts:
                                    alerts.append(clean_line)
                                    
                        # Keep alerts trim
                        if len(alerts) > 10:
                            alerts = alerts[-10:]
            
            # 3. Render
            render_dashboard(stats, latest_heartbeat, alerts)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n{BOLD}[*] Monitor gracefully shut down.{RESET}")

if __name__ == "__main__":
    run_monitor()
