import subprocess
import time
import os

def launch_experiment():
    print("="*60)
    print("ANTIGRAVITY LINEAR PROTOCOL: 0.1B PARAMETER PROJECT")
    print("="*60)

    # Clean previous comparison files
    files_to_clean = [
        "stats_project_real.json", "samples_project_real.json", "perplexity_project_real.json",
        "project_real_live.pt", "training_project_real_0.1b.log"
    ]
    # for f in files_to_clean:
    #     if os.path.exists(f):
    #         os.remove(f)

    def start_proc(name, manifold, mode, stats, prefix, dim, n_layers, n_heads):
        print(f"[*] Launching {name.upper()} session ({mode}) | {dim}d {n_layers}L {n_heads}H...")
        return subprocess.Popen([
            "python", "-u", "primal_train_modular.py",
            "--manifold", manifold,
            "--mode", mode,
            "--stats_file", stats,
            "--checkpoint_prefix", prefix,
            "--dim", str(dim),
            "--n_layers", str(n_layers),
            "--n_heads", str(n_heads)
        ], stdout=open(f"training_{name}.log", "w"), stderr=subprocess.STDOUT)

    # Project Real: Recursive-Refinement v4.0
    # Virtual 16-layer reasoning engine
    procs = {
        "project_real": start_proc("project_real_v4", "linear", "primal", "stats_project_real.json", "project_real", 768, 8, 0)
    }

    print("[*] Project Real: 0.1B Linear LLM Launched.")
    
    try:
        while True:
            time.sleep(10)
            for name, p in procs.items():
                if p.poll() is not None:
                    print(f"[!] {name.upper()} session terminated.")
                    return
    except KeyboardInterrupt:
        print("\n[*] Stopping all experiments...")
        for p in procs.values():
            p.terminate()

if __name__ == "__main__":
    launch_experiment()
