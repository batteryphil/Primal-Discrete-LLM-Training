import soundfile as sf
import numpy as np
import sys

def check_wav(path):
    try:
        data, samplerate = sf.read(path)
        print(f"Path: {path}")
        print(f"Sample Rate: {samplerate}")
        print(f"Duration: {len(data)/samplerate:.2f}s")
        print(f"Max Amplitude: {np.max(np.abs(data))}")
        print(f"Mean Amplitude: {np.mean(np.abs(data))}")
        if np.max(np.abs(data)) < 1e-5:
            print("[!] FILE IS SILENT")
        else:
            print("[*] File contains audio data")
    except Exception as e:
        print(f"[!] Error reading file: {e}")

if __name__ == "__main__":
    check_wav(sys.argv[1])
