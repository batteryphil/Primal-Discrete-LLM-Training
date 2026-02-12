import time
import os
import sys
import re

LOG_FILE = "training.log"

def monitor():
    print(f"[*] Monitoring {LOG_FILE}...")
    if not os.path.exists(LOG_FILE):
        print(f"[!] {LOG_FILE} not found. Waiting for training to start...")
        while not os.path.exists(LOG_FILE):
            time.sleep(1)
    
    with open(LOG_FILE, "r", encoding='utf-8', errors='replace') as f:
        # Go to the end of the file
        f.seek(0, os.SEEK_END)
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            # Print raw line
            print(line, end="")
            
            # Simple parsing for highlighting
            if "Step" in line and "Loss" in line:
                # You could add color here if you wanted
                pass

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n[*] Monitoring stopped.")
