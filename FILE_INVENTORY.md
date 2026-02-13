# Trinity-1.58bit File Inventory

> **Master list of all project files.** Files marked as ARCHIVED were removed from the local project during cleanup but are preserved in the git history. To recover any archived file, run:
> ```
> git checkout 3d0e451 -- <filename>
> ```

## 🟢 Active Files (Local)

### Core Scripts
| File | Purpose |
|---|---|
| `primal_train_ghost.py` | Primary training script (Phase 60) |
| `manifolds.py` | 8-bit Prime-Harmonic LUT generator |
| `primal_val_perplexity.py` | WikiText-2 perplexity validation |
| `primal_bench.py` | Automated benchmark suite |
| `primal_infer_ghost.py` | Ghost inference / generation |
| `monitor_auto_bench.py` | Auto-benchmark monitor |

### Configuration & Launchers
| File | Purpose |
|---|---|
| `run_ghost.bat` | Training launcher script |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |
| `.gitattributes` | Git LFS / attributes |

### Documentation
| File | Purpose |
|---|---|
| `README.md` | Project README |
| `LICENSE` | MIT License |
| `PAPER.md` | Technical paper draft |
| `BENCHMARKS.md` | Benchmark results |
| `ROADMAP.md` | Development roadmap |
| `FILE_INVENTORY.md` | This file |
| `project_primal_v3_source.txt` | Active source documentation |

### Model Artifacts (gitignored)
| File | Size | Purpose |
|---|---|---|
| `primal_ghost_live.pt` | 579MB | **Active checkpoint** |
| `training.log` | ~95KB | Current training log |
| `manifold_step_100.png` | 25KB | Latest manifold heatmap |

### C++ Inference Engine (`src/`)
| File | Purpose |
|---|---|
| `src/__init__.py` | Package init |
| `src/pack.py` | Model packer |
| `src/run_inference.py` | Python inference runner |
| `src/train.py` | Training utilities |
| `src/engine/` | Engine module |
| `src/speed_run/trinity_loader.cpp` | C++ model loader |

### Compiled Engine Binaries (`src/engine/`)
| File | Purpose |
|---|---|
| `src/engine/Makefile` | Build system |
| `src/engine/build.bat` | Windows build script |
| `src/engine/main.cpp` | Entry point |
| `src/engine/primal.cpp` / `primal.h` | Core runtime |
| `src/engine/primal_lib.cpp` | Library interface |
| `src/engine/cpu_kernel.cpp` | CPU inference kernel |
| `src/engine/gpu_kernel.cu` | CUDA inference kernel |
| `src/engine/gpu_dummy.cpp` | GPU fallback |
| `src/engine/trinity_cpu.exe` | CPU inference binary |
| `src/engine/trinity_gpu.exe` | GPU inference binary |
| `src/engine/*.obj`, `*.exp`, `*.lib` | Build artifacts |
| `src/engine/model.primal` | Engine model copy |

### Packed Models (`models/`)
| File | Size | Purpose |
|---|---|---|
| `models/trinity_1.58bit_packed.bin` | 258MB | Packed binary model |

---

## 🔴 Archived Files (Removed Locally — In Git History)

> Recovery: `git checkout 3d0e451 -- <filename>`

### Legacy Training Scripts
| File | Size | Original Purpose |
|---|---|---|
| `primal_train.py` | 12KB | Original training script |
| `primal_train_v2.py` | 6KB | V2 training script |
| `primal_train_v3.py` | 11KB | V3 training script |
| `primal_train_jit.py` | 10KB | JIT compilation experiment |

### Legacy Utility Scripts
| File | Size | Original Purpose |
|---|---|---|
| `primal_export_v4.py` | 2.5KB | Model export tool |
| `primal_packer.py` | 3KB | Model packer (superseded by `src/pack.py`) |
| `primal_chat.py` | 3.5KB | Chat interface |
| `primal_qa.py` | 1KB | QA evaluation |
| `monitor_primal.py` | 1KB | Old training monitor |
| `scripts/verify_launch.py` | 772B | One-time verification |

### Legacy Source Snapshots
| File | Size | Original Purpose |
|---|---|---|
| `project_trinity_full_source.txt` | 39KB | Full source (outdated) |
| `project_trinity_v2_source.txt` | 8KB | V2 source snapshot |
| `project_trinity_v3_source.txt` | 6KB | V3 source snapshot |

### Legacy Training Logs
| File | Size | Phase |
|---|---|---|
| `debug.log` | 6KB | Debug output |
| `ghost_train_aligned.log` | 5KB | Aligned training |
| `ghost_train_greedy.log` | 66KB | Greedy training |
| `ghost_train_poltergeist.log` | 156KB | Poltergeist v1 |
| `ghost_train_poltergeist_v2.log` | 98KB | Poltergeist v2 |
| `ghost_training.log` | 132KB | General training |
| `ghost_training_adf.log` | 68KB | ADF training |
| `ghost_training_phase38.log` | 2.5MB | Phase 38 (largest) |
| `ghost_training_phase39.log` | 89KB | Phase 39 |
| `ghost_training_phase40.log` | 121KB | Phase 40 |
| `ghost_training_phase40_fine.log` | 47KB | Phase 40 fine-tuning |
| `ghost_training_salad.log` | 303KB | Salad test logs |
| `ghost_training_stabilized.log` | 72KB | Stabilization |
| `ghost_training_utf8.log` | 4KB | UTF-8 fix |
| `ghost_training_utf8_final.log` | 40KB | UTF-8 final |
| `ghost_training_v4.log` | 11KB | V4 training |

### Legacy Announcements
| File | Size | Original Purpose |
|---|---|---|
| `DISCORD_ANNOUNCEMENT.md` | 2.6KB | Discord announcement |
| `MARKETING.md` | 1.4KB | Marketing draft |
| `PR_ANNOUNCEMENT.txt` | 2.6KB | PR announcement |

### Legacy Batch/Build Files
| File | Size | Original Purpose |
|---|---|---|
| `run_nitrous.bat` | 206B | Old nitrous launcher |
| `run_test.bat` | 358B | Old test runner |
| `build_wrapper.bat` | 134B | Old build wrapper |
| `Makefile-SpeedRun` | 128B | Duplicate Makefile |

### Legacy Data Files
| File | Size | Original Purpose |
|---|---|---|
| `ppl_result.txt` | 4KB | Old perplexity result |
| `ppl_result_11600.txt` | 4KB | Step 11600 perplexity |

---

## 🟡 Large Gitignored Files (Deleted Locally — NOT in git)

> ⚠️ These files were gitignored and are NOT recoverable from git history.

| File | Size | Notes |
|---|---|---|
| `primal_ghost_step16398.pt` | 745MB | Phase 38 fallback checkpoint |
| `primal_origin.pt` | 137MB | Original pre-training weights |
| `primal_expanse.pt` | 253MB | Legacy expanded model |
| `primal_expanse.primal` | 20MB | Legacy packed model |
| `model.primal` | 55MB | Legacy packed model |
| `primal_trained.primal` | 9.5MB | Legacy packed model |
