import tkinter as tk
from tkinter import ttk, font
import json
import os
import time
import math

class PrimalMonitorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("PRIMAL Engine - Night Shift Monitor")
        self.geometry("1100x800")
        self.configure(bg="#0d0d12")
        self.minsize(900, 600)
        
        # UI Colors (Cyber/Terminal Theme)
        self.c_bg = "#0d0d12"
        self.c_panel = "#15151e"
        self.c_text = "#8b8b9f"
        self.c_data = "#e2e2ec"
        self.c_accent = "#ff2a6d"  # Neon pink/red for primal theme
        self.c_acc2 = "#05d9e8"    # Cyan
        self.c_acc3 = "#01ffe5"    # Bright cyan
        self.c_warn = "#ffc107"

        # Fonts
        self.f_title = font.Font(family="Consolas", size=24, weight="bold")
        self.f_sub = font.Font(family="Consolas", size=14, weight="bold")
        self.f_data = font.Font(family="Consolas", size=32, weight="bold")
        self.f_text = font.Font(family="Consolas", size=10)
        
        self.init_ui()
        
        # Polling
        self.run_updates()

    def init_ui(self):
        # Header
        header = tk.Frame(self, bg=self.c_bg, pady=15)
        header.pack(fill=tk.X)
        
        lbl_title = tk.Label(header, text="/// PRIMAL NIGHT SHIFT SUPERVISOR V7", 
                             font=self.f_title, bg=self.c_bg, fg=self.c_accent)
        lbl_title.pack(side=tk.LEFT, padx=20)
        
        self.lbl_status = tk.Label(header, text="● ACTIVE", font=self.f_sub,
                                   bg=self.c_bg, fg=self.c_acc3)
        self.lbl_status.pack(side=tk.RIGHT, padx=20)
        
        # Main Layout
        content = tk.Frame(self, bg=self.c_bg)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Top Row: Big Stats
        top_row = tk.Frame(content, bg=self.c_bg)
        top_row.pack(fill=tk.X, pady=(0, 20))
        
        self.v_step = self.create_stat_card(top_row, "CURRENT STEP", "0")
        self.v_loss = self.create_stat_card(top_row, "MANIFOLD LOSS", "0.0000")
        self.v_tps = self.create_stat_card(top_row, "THROUGHPUT (TPS)", "0.0")
        self.v_flips = self.create_stat_card(top_row, "VOTER FLIPS (M)", "0.0000")
        
        # Middle Row: Graph Canvas
        mid_row = tk.Frame(content, bg=self.c_panel, bd=1, relief=tk.SOLID)
        mid_row.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        lbl_graph = tk.Label(mid_row, text="LOSS TRAJECTORY", font=self.f_sub, bg=self.c_panel, fg=self.c_text)
        lbl_graph.pack(anchor=tk.W, padx=10, pady=10)
        
        self.canvas = tk.Canvas(mid_row, bg=self.c_panel, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", lambda e: self.draw_graph())
        
        # Bottom Row: Salad Output
        bot_row = tk.Frame(content, bg=self.c_panel, bd=1, relief=tk.SOLID)
        bot_row.pack(fill=tk.X, ipady=10)
        
        lbl_salad = tk.Label(bot_row, text="STRUCTURAL HEALING PATTERN (WORD SALAD)", font=self.f_sub, bg=self.c_panel, fg=self.c_text)
        lbl_salad.pack(anchor=tk.W, padx=10, pady=5)
        
        self.txt_salad = tk.Text(bot_row, height=8, bg=self.c_bg, fg=self.c_data, 
                                 font=self.f_text, relief=tk.FLAT, wrap=tk.WORD)
        self.txt_salad.pack(fill=tk.X, padx=10, pady=5)
        
        self.last_stats = []
        self.last_salad = ""
        
    def create_stat_card(self, parent, title, default_val):
        card = tk.Frame(parent, bg=self.c_panel, bd=1, relief=tk.SOLID)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        lbl_t = tk.Label(card, text=title, font=self.f_text, bg=self.c_panel, fg=self.c_text)
        lbl_t.pack(anchor=tk.NW, padx=10, pady=(10, 0))
        
        lbl_v = tk.Label(card, text=default_val, font=self.f_data, bg=self.c_panel, fg=self.c_data)
        lbl_v.pack(anchor=tk.NE, padx=10, pady=(0, 10))
        
        return lbl_v

    def run_updates(self):
        self.fetch_data()
        self.draw_graph()
        self.after(2000, self.run_updates) # 2 second polling

    def fetch_data(self):
        try:
            if os.path.exists("stats_coder.json"):
                with open("stats_coder.json", "r") as f:
                    history = json.load(f)
                    self.last_stats = history
                    if history:
                        latest = history[-1]
                        self.v_step.config(text=f"{latest.get('step', 0)}")
                        self.v_loss.config(text=f"{latest.get('loss', 0.0):.4f}")
                        self.v_tps.config(text=f"{latest.get('tps', 0.0):.1f}")
                        self.v_flips.config(text=f"{latest.get('flips', 0.0):.4f}")
                        
                        # Check activity (Data older than 60s is stale)
                        time_diff = time.time() - latest.get("timestamp", time.time())
                        if time_diff > 60:
                            self.lbl_status.config(text="● IDLE/STALLED", fg=self.c_warn)
                        else:
                            self.lbl_status.config(text="● ACTIVE", fg=self.c_acc3)
                            
        except Exception as e:
            pass

        try:
             if os.path.exists("samples_coder.json"):
                with open("samples_coder.json", "r") as f:
                    salads = json.load(f)
                    if salads:
                        latest_salad = salads[-1]
                        text = f"[Step {latest_salad.get('step', '?')}]\n{latest_salad.get('text', '')}"
                        if text != self.last_salad:
                            self.txt_salad.delete(1.0, tk.END)
                            self.txt_salad.insert(tk.END, text)
                            self.last_salad = text
        except Exception:
            pass

    def draw_graph(self):
        self.canvas.delete("all")
        if not self.last_stats or len(self.last_stats) < 2:
            return
            
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10: return
        
        # Add padding
        pad_x = 60
        pad_y = 30
        plt_w = w - (pad_x * 2)
        plt_h = h - (pad_y * 2)
        
        # We plot all points to keep the curve continually expanding
        pts = self.last_stats
        
        losses = [p.get("loss", 0.0) for p in pts]
        min_l = min(losses)
        max_l = max(losses)
        diff = max_l - min_l
        if diff == 0: diff = 1.0 # avoid div by 0
        
        # Margin for graph min/max
        min_l -= diff * 0.1
        max_l += diff * 0.1
        diff = max_l - min_l
        
        steps = [p.get("step", 0.0) for p in pts]
        min_s = steps[0]
        max_s = steps[-1]
        diff_s = max_s - min_s
        if diff_s == 0: diff_s = 1.0
        
        # Draw Grid and Labels
        for i in range(5):
            y_pos = pad_y + (i * plt_h / 4)
            loss_val = max_l - (diff * (i / 4.0))
            self.canvas.create_line(pad_x, y_pos, w - pad_x, y_pos, fill="#2a2a35", width=1, dash=(4, 4))
            self.canvas.create_text(pad_x - 15, y_pos, text=f"{loss_val:.2f}", fill=self.c_text, font=self.f_text, anchor=tk.E)
            
        for i in range(5):
            x_pos = pad_x + (i * plt_w / 4)
            step_val = min_s + (diff_s * (i / 4.0))
            self.canvas.create_line(x_pos, pad_y, x_pos, h - pad_y, fill="#2a2a35", width=1, dash=(4, 4))
            self.canvas.create_text(x_pos, h - pad_y + 15, text=f"{int(step_val)}", fill=self.c_text, font=self.f_text, anchor=tk.N)
        
        # Draw line
        coords = []
        for p in pts:
            x = pad_x + ((p.get("step", 0) - min_s) / diff_s) * plt_w
            y = pad_y + plt_h - ((p.get("loss", 0) - min_l) / diff) * plt_h
            coords.extend([x, y])
            
        if len(coords) >= 4:
            poly_coords = [pad_x, h - pad_y] + coords + [coords[-2], h - pad_y]
            self.canvas.create_polygon(poly_coords, fill="#0a1a2a", outline="")
            
            # Glow effect (stacking smooth cyan lines)
            self.canvas.create_line(coords, fill="#04595e", width=8, smooth=True)
            self.canvas.create_line(coords, fill="#058b94", width=5, smooth=True)

            # Main crisp line
            self.canvas.create_line(coords, fill=self.c_acc3, width=2, smooth=True)
            
            # Data points
            for i in range(0, len(coords), 2):
                x, y = coords[i], coords[i+1]
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill=self.c_bg, outline=self.c_acc3, width=2)

if __name__ == "__main__":
    app = PrimalMonitorGUI()
    app.mainloop()
