
"""
Moss Seed — Gradio Simulator (v0.2)
Author: Nova (for Kumail / AI Cadmey)

Run:
  pip install gradio numpy matplotlib
  python moss_seed_gradio.py

What you get:
- Live grid of moss biomass (seed at center).
- Preset environments + manual sliders (humidity, light, temp, pH, nutrients).
- Wet–dry cycles (amplitude/period) to mimic intermittent rain.
- Start / Pause / Step / Reset buttons.
- Realtime stats: step, coverage, mean biomass.

Notes:
- This is a pedagogical model (cellular-ish automaton + suitability-driven growth).
- No seaborn used; simple numpy + matplotlib colormap for RGB image.
"""

import numpy as np
import matplotlib.cm as cm
import gradio as gr

# ------------------------ Model Parameters ------------------------

GRID_SIZE = 960              # grid cells per side
INIT_BIOMASS = 0.30         # initial biomass for the seed (0..1)
MAX_BIOMASS = 1.0           # cap
DECAY_RATE = 0.06           # per step decay when unsuitable
SPREAD_BASE = 0.20          # base colonization chance (scaled by suitability and neighbors)
NEIGHBOR_WEIGHT = 0.35      # neighbor biomass -> colonization boost
NOISE = 0.02                # stochasticity

# Moss "genotype" (optima + tolerances)
MOSS_OPTIMA = dict(
    humidity=0.75,       # 0..1
    light=0.45,          # 0..1
    temperature=12.0,    # °C
    pH=5.5,              # acidic favored (e.g., many wall/peat mosses)
    nutrients=0.35       # 0..1 (thrive in low-moderate nutrients)
)
MOSS_TOL = dict(           # larger = broader tolerance
    humidity=0.25,
    light=0.35,
    temperature=10.0,
    pH=1.5,
    nutrients=0.30
)

# Environment presets (illustrative; tweak freely)
PRESETS = {
    "Urban Wall (shaded, intermittent wet)": dict(humidity=0.55, light=0.35, temperature=14.0, pH=7.2, nutrients=0.30),
    "Temperate Forest Floor":                dict(humidity=0.80, light=0.30, temperature=10.0, pH=5.5, nutrients=0.35),
    "Boreal/Arctic Tundra":                  dict(humidity=0.60, light=0.55, temperature=2.0,  pH=5.0, nutrients=0.25),
    "Desert Rock (ephemeral rain)":          dict(humidity=0.25, light=0.80, temperature=28.0, pH=7.8, nutrients=0.15),
    "Rainforest Trunk (humid & shaded)":     dict(humidity=0.95, light=0.25, temperature=22.0, pH=5.2, nutrients=0.45),
}

# ------------------------ Core Simulation ------------------------

def logistic(x):  # stable logistic
    return 1.0 / (1.0 + np.exp(-x))

def suitability_score(env):
    """Raw suitability score (higher is better)."""
    d_h = abs(env['humidity']    - MOSS_OPTIMA['humidity'])    / (MOSS_TOL['humidity'] + 1e-6)
    d_l = abs(env['light']       - MOSS_OPTIMA['light'])       / (MOSS_TOL['light'] + 1e-6)
    d_t = abs(env['temperature'] - MOSS_OPTIMA['temperature']) / (MOSS_TOL['temperature'] + 1e-6)
    d_p = abs(env['pH']          - MOSS_OPTIMA['pH'])          / (MOSS_TOL['pH'] + 1e-6)
    d_n = abs(env['nutrients']   - MOSS_OPTIMA['nutrients'])   / (MOSS_TOL['nutrients'] + 1e-6)

    penalty = 1.1*d_h + 1.0*d_l + 1.0*d_t + 0.9*d_p + 0.8*d_n
    return 2.0 - penalty  # ~0 near optimum; negative as distance grows

def moisture_pulse(step, base_humidity, period=30, amplitude=0.20):
    """Sinusoidal wet–dry cycles around base humidity."""
    return float(np.clip(base_humidity + amplitude * np.sin(2*np.pi*step/max(2,period)), 0.0, 1.0))

def neighborhood_biomass(grid):
    """Average of 8-neighbor biomass (wraparound)."""
    shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    acc = np.zeros_like(grid)
    for dx,dy in shifts:
        acc += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    return acc / 8.0

class MossSim:
    def __init__(self):
        self.env = PRESETS["Urban Wall (shaded, intermittent wet)"].copy()
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        self.center_seed()
        self.step_id = 0
        self.running = False
        self.cycle_on = True
        self.cycle_period = 26
        self.cycle_amplitude = 0.22

    def center_seed(self):
        c = GRID_SIZE // 2
        self.grid[c, c] = INIT_BIOMASS

    def set_env_dict(self, env_dict):
        self.env = env_dict.copy()

    def set_env_values(self, humidity, light, temperature, pH, nutrients):
        self.env = dict(humidity=float(humidity), light=float(light),
                        temperature=float(temperature), pH=float(pH), nutrients=float(nutrients))

    def reset(self):
        self.grid[...] = 0.0
        self.center_seed()
        self.step_id = 0
        self.running = False

    def stats(self):
        coverage = float((self.grid > 0.05).mean())
        nz = self.grid[self.grid > 0.0]
        mean_b = 0.0 if nz.size == 0 else float(nz.mean())
        return coverage, mean_b

    def step(self):
        self.step_id += 1

        eff_env = self.env.copy()
        if self.cycle_on:
            eff_env['humidity'] = moisture_pulse(self.step_id, self.env['humidity'],
                                                 period=self.cycle_period, amplitude=self.cycle_amplitude)
        # Suitability -> [0,1]
        suit = logistic(suitability_score(eff_env))

        neigh = neighborhood_biomass(self.grid)

        colonize_prob = SPREAD_BASE * suit * (1.0 + NEIGHBOR_WEIGHT * neigh)
        colonize_prob = np.clip(colonize_prob, 0.0, 1.0)

        growth = 0.10 * suit * (1.0 + 0.5 * neigh)
        decay = DECAY_RATE * (1.0 - suit)

        rand = np.random.rand(GRID_SIZE, GRID_SIZE)

        new_grid = self.grid.copy()

        colonize_mask = (self.grid < 0.02) & (rand < colonize_prob)
        if colonize_mask.any():
            new_grid[colonize_mask] = INIT_BIOMASS * (0.8 + 0.4*np.random.rand(np.sum(colonize_mask)))

        grow_mask = self.grid >= 0.02
        new_grid[grow_mask] = np.minimum(MAX_BIOMASS, new_grid[grow_mask] + growth[grow_mask])

        poor_mask = suit < 0.45
        new_grid[poor_mask] = np.maximum(0.0, new_grid[poor_mask] - decay)

        new_grid += NOISE * (np.random.rand(GRID_SIZE, GRID_SIZE) - 0.5)
        self.grid = np.clip(new_grid, 0.0, MAX_BIOMASS)

# ------------------------ Rendering ------------------------

def grid_to_rgb(grid):
    """Convert [0,1] grid to RGB using matplotlib's viridis colormap."""
    cmap = cm.get_cmap('viridis')
    rgba = cmap(np.clip(grid, 0.0, 1.0))
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb

# ------------------------ Gradio Wiring ------------------------

SIM = MossSim()  # single sim instance

def do_start():
    SIM.running = True
    cov, mb = SIM.stats()
    return f"▶ Running | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def do_pause_resume():
    SIM.running = not SIM.running
    cov, mb = SIM.stats()
    state = "▶ Running" if SIM.running else "⏸ Paused"
    return f"{state} | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def do_step():
    was = SIM.running
    SIM.running = False
    SIM.step()
    img = grid_to_rgb(SIM.grid)
    cov, mb = SIM.stats()
    SIM.running = was
    return img, f"Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def do_reset():
    SIM.reset()
    img = grid_to_rgb(SIM.grid)
    return img, "Reset complete. Ready."

def apply_preset(preset_name):
    env = PRESETS[preset_name]
    SIM.set_env_dict(env)
    # return slider values + status text
    cov, mb = SIM.stats()
    status = f"Preset: {preset_name} | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"
    return env['humidity'], env['light'], env['temperature'], env['pH'], env['nutrients'], status

def set_env_from_sliders(h, l, t, pH, n):
    SIM.set_env_values(h, l, t, pH, n)
    cov, mb = SIM.stats()
    return f"Updated env | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def set_cycle(on, period, amplitude):
    SIM.cycle_on = bool(on)
    SIM.cycle_period = int(period)
    SIM.cycle_amplitude = float(amplitude)
    cov, mb = SIM.stats()
    return f"Cycles: {'ON' if SIM.cycle_on else 'OFF'} (period={SIM.cycle_period}, amp={SIM.cycle_amplitude:.2f}) | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def tick():
    """Timer tick — advance if running and return image + status."""
    if SIM.running:
        SIM.step()
    img = grid_to_rgb(SIM.grid)
    cov, mb = SIM.stats()
    hum = SIM.env['humidity']
    status = (f"{'▶ Running' if SIM.running else '⏸ Paused'} | Step {SIM.step_id} | "
              f"Coverage {cov:.2f} | Mean biomass {mb:.2f} | "
              f"Env: H={hum:.2f}{'~' if SIM.cycle_on else ''}, L={SIM.env['light']:.2f}, "
              f"T={SIM.env['temperature']:.1f}°C, pH={SIM.env['pH']:.1f}, N={SIM.env['nutrients']:.2f}")
    return img, status

with gr.Blocks(title="Moss Seed — Real‑Time Simulator") as demo:
    gr.Markdown("# 🌱 Moss Seed — Real‑Time Growth / Decomposition Simulator")
    gr.Markdown(
        "Seed starts at the center. Choose an environment or tweak sliders, then **Start**. "
        "Switch conditions mid-run to watch adaptation or decay.\n"
        "_Pedagogical model for research exploration — tune freely._"
    )

    with gr.Row():
        canvas = gr.Image(label="Moss Biomass (0–1)", image_mode="RGB", interactive=False, height=520)
    with gr.Row():
        start_btn = gr.Button("Start", variant="primary")
        pause_btn = gr.Button("Pause/Resume")
        step_btn  = gr.Button("Step")
        reset_btn = gr.Button("Reset")

    with gr.Row():
        preset = gr.Dropdown(choices=list(PRESETS.keys()), value="Urban Wall (shaded, intermittent wet)",
                             label="Environment Preset")

    with gr.Row():
        humidity = gr.Slider(0, 1, value=PRESETS[preset.value]["humidity"], step=0.01, label="Humidity")
        light    = gr.Slider(0, 1, value=PRESETS[preset.value]["light"], step=0.01, label="Light")
        temp     = gr.Slider(-10, 40, value=PRESETS[preset.value]["temperature"], step=0.5, label="Temperature (°C)")
        ph       = gr.Slider(3.5, 9.0, value=PRESETS[preset.value]["pH"], step=0.1, label="pH")
        nutr     = gr.Slider(0, 1, value=PRESETS[preset.value]["nutrients"], step=0.01, label="Nutrients")

    with gr.Row():
        cyc_on   = gr.Checkbox(value=True, label="Wet–Dry Cycles ON")
        cyc_per  = gr.Slider(5, 120, value=26, step=1, label="Cycle Period (steps)")
        cyc_amp  = gr.Slider(0.0, 0.5, value=0.22, step=0.01, label="Cycle Amplitude")

    status = gr.Textbox(label="Status / Telemetry", value="Ready.", interactive=False)

    # Initial image render
    canvas.value = grid_to_rgb(SIM.grid)

    # Button events
    start_btn.click(fn=do_start, outputs=status)
    pause_btn.click(fn=do_pause_resume, outputs=status)
    step_btn.click(fn=do_step, outputs=[canvas, status])
    reset_btn.click(fn=do_reset, outputs=[canvas, status])

    # Preset change -> set env + sliders + status
    preset.change(fn=apply_preset, inputs=preset, outputs=[humidity, light, temp, ph, nutr, status])

    # Slider changes -> update env
    for ctrl in (humidity, light, temp, ph, nutr):
        ctrl.change(fn=set_env_from_sliders, inputs=[humidity, light, temp, ph, nutr], outputs=status)

    # Cycle controls
    cyc_on.change(fn=set_cycle, inputs=[cyc_on, cyc_per, cyc_amp], outputs=status)
    cyc_per.change(fn=set_cycle, inputs=[cyc_on, cyc_per, cyc_amp], outputs=status)
    cyc_amp.change(fn=set_cycle, inputs=[cyc_on, cyc_per, cyc_amp], outputs=status)

    # Timer for live updates
    demo.load(fn=tick, outputs=[canvas, status])  # do one render on load
    gr.Timer(0.15).tick(fn=tick, outputs=[canvas, status])  # ~6-7 FPS

if __name__ == "__main__":
    demo.launch()
