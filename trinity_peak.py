import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Trinity Peak Monitoring System")

STATS_FILE = "stats.json"
LOG_FILE = "training_qwen.log"

class StatItem(BaseModel):
    step: int
    loss: float
    flips: float
    tps: float
    pscale: float
    anneal: float
    vram: float
    timestamp: float

import time

def read_json_safe(filepath, retries=5, delay=0.1):
    """Robustly read a JSON file, retrying on failure or empty content."""
    for i in range(retries):
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    content = f.read().strip()
                    if not content:
                        raise ValueError("Empty file")
                    return json.loads(content)
        except (json.JSONDecodeError, ValueError, OSError):
            time.sleep(delay)
    return []

@app.get("/api/stats/v5")
async def get_v5_stats():
    data = read_json_safe("stats_v5.json")
    if data:
        clean_data = json.loads(json.dumps(data).replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null'))
        return clean_data
    return []

@app.get("/api/samples/v5")
async def get_v5_samples():
    return read_json_safe("samples_v5.json")

@app.get("/api/perplexity/v5")
async def get_v5_perplexity():
    return read_json_safe("perplexity_v5.json")

@app.get("/api/stats/project_real")
async def get_project_real_stats():
    data = read_json_safe("stats_coder.json")
    if data:
        # Sanitize NaN/Inf for JSON compliance
        clean_data = json.loads(json.dumps(data).replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null'))
        return clean_data
    return []

@app.get("/api/samples/project_real")
async def get_project_real_samples():
    return read_json_safe("samples_coder.json")

@app.get("/api/perplexity/project_real")
async def get_project_real_perplexity():
    return read_json_safe("perplexity_coder.json")

@app.get("/api/voter_map")
async def get_voter_map():
    return read_json_safe("voter_map_stats.json")

@app.get("/api/supervisor")
async def get_supervisor_state():
    if os.path.exists("supervisor_state.json"):
        try:
            with open("supervisor_state.json", "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # Catch mid-write read collisions or file locking issues
            pass 
            
    # Fallback default if the file hasn't been written yet or is temporarily locked
    return {
        "current_threshold": 20, 
        "min_thresh": 17, 
        "max_thresh": 24, 
        "starvation_patience": 0
    }

@app.get("/api/recovery_status")
async def get_recovery_status():
    """Returns the current status of the discrete healing mission."""
    last_step = 0
    if os.path.exists("stats_coder.json"):
        try:
            with open("stats_coder.json", "r") as f:
                data = json.load(f)
                if data: last_step = data[-1]['step']
        except: pass
    
    if last_step < 200:
        phase = "Stabilization"
        detail = "Fixing Language Head & Embeddings"
        peel = "Locked"
    else:
        phase = "Layer Peel"
        unlocked = min(((last_step - 200) // 50 + 1) * 2, 28)
        detail = f"Harmonic healing active for top {unlocked} layers"
        peel = f"Unlocked [{28-unlocked}-27]"

    return {
        "step": last_step,
        "phase": phase,
        "detail": detail,
        "peel": peel,
        "target": "Qwen2.5-Coder-1.5B PRIME"
    }

@app.get("/api/stats")
async def get_stats():
    data = read_json_safe("stats_coder.json")
    if data:
        # Sanitize NaN/Inf for JSON compliance
        clean_data = json.loads(json.dumps(data).replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null'))
        return clean_data
    return []

@app.get("/api/logs")
async def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": "Log file not found."}
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return {"logs": "".join(lines[-100:])}
    except Exception as e:
        return {"logs": f"Error: {str(e)}"}

# Serve static files from dashboard directory
app.mount("/", StaticFiles(directory="dashboard", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
