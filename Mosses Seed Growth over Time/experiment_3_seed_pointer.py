
"""
Moss Seed — Spatially Aware Simulator (v0.6)
Author: Nova (for Kumail / AI Cadmey)

New in v0.6
- ✅ Click-to-seed: click anywhere on the canvas to plant a new seed.
- ✅ Speed slider: choose how many simulation STEPS happen per timer tick.

Run:
  pip install gradio numpy matplotlib
  python moss_seed_gradio_spatial_v0_6.py
"""

import numpy as np
import gradio as gr
import matplotlib.cm as cm

# ------------------------ Base Parameters ------------------------

GRID_SIZE = 1000
INIT_BIOMASS = 0.30
MAX_BIOMASS = 1.0
DECAY_RATE = 0.06
SPREAD_BASE = 0.20
NEIGHBOR_WEIGHT = 0.35
NOISE = 0.02
OCC_THR = 0.05  # occupied if biomass >= OCC_THR

MOSS_OPTIMA = dict(humidity=0.75, light=0.45, temperature=12.0, pH=5.5, nutrients=0.35)
MOSS_TOL    = dict(humidity=0.25, light=0.35, temperature=10.0, pH=1.5, nutrients=0.30)

PRESETS = {
    "Urban Wall (shaded, intermittent wet)": dict(humidity=0.55, light=0.35, temperature=14.0, pH=7.2, nutrients=0.30),
    "Temperate Forest Floor":                dict(humidity=0.80, light=0.30, temperature=10.0, pH=5.5, nutrients=0.35),
    "Boreal/Arctic Tundra":                  dict(humidity=0.60, light=0.55, temperature=2.0,  pH=5.0, nutrients=0.25),
    "Desert Rock (ephemeral rain)":          dict(humidity=0.25, light=0.80, temperature=28.0, pH=7.8, nutrients=0.15),
    "Rainforest Trunk (humid & shaded)":     dict(humidity=0.95, light=0.25, temperature=22.0, pH=5.2, nutrients=0.45),
}
DEFAULT_PRESET = "Urban Wall (shaded, intermittent wet)"

DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
DIR_VECS = np.array(DIRS, dtype=float)
DIR_VECS /= np.linalg.norm(DIR_VECS, axis=1, keepdims=True)

# ------------------------ Utilities ------------------------

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def suitability_raw(env):
    d_h = abs(env['humidity']    - MOSS_OPTIMA['humidity'])    / (MOSS_TOL['humidity'] + 1e-6)
    d_l = abs(env['light']       - MOSS_OPTIMA['light'])       / (MOSS_TOL['light'] + 1e-6)
    d_t = abs(env['temperature'] - MOSS_OPTIMA['temperature']) / (MOSS_TOL['temperature'] + 1e-6)
    d_p = abs(env['pH']          - MOSS_OPTIMA['pH'])          / (MOSS_TOL['pH'] + 1e-6)
    d_n = abs(env['nutrients']   - MOSS_OPTIMA['nutrients'])   / (MOSS_TOL['nutrients'] + 1e-6)
    penalty = 1.1*d_h + 1.0*d_l + 1.0*d_t + 0.9*d_p + 0.8*d_n
    return 2.0 - penalty

def neighborhood_mean(arr):
    acc = np.zeros_like(arr)
    for dx,dy in DIRS:
        acc += np.roll(np.roll(arr, dx, axis=0), dy, axis=1)
    return acc / 8.0

def moisture_pulse(step, base_h, period=30, amp=0.20):
    return float(np.clip(base_h + amp*np.sin(2*np.pi*step/max(2,period)), 0.0, 1.0))

def make_light_field(base_light, grad_strength, sun_deg):
    H, W = GRID_SIZE, GRID_SIZE
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y)
    th = np.deg2rad(sun_deg)
    vx, vy = np.cos(th), np.sin(th)
    g = grad_strength * (vx*X + vy*Y)
    L = np.clip(base_light + g, 0.0, 1.0)
    return L

def apply_shade(light_field, biomass, shade_coeff):
    shade = neighborhood_mean(biomass)
    L_eff = np.clip(light_field * (1.0 - shade_coeff*shade), 0.0, 1.0)
    return L_eff

def diffuse(arr, k):
    if k <= 0:
        return arr
    up    = np.roll(arr, -1, axis=0)
    down  = np.roll(arr, 1, axis=0)
    left  = np.roll(arr, -1, axis=1)
    right = np.roll(arr, 1, axis=1)
    lap = (up + down + left + right - 4*arr)
    return np.clip(arr + k * lap, 0.0, 1.0)

def advect(arr, wind_deg, alpha):
    if alpha <= 0:
        return arr
    th = np.deg2rad(wind_deg % 360)
    dx = int(round(np.sin(th)))   # y axis downwards
    dy = int(round(np.cos(th)))
    shifted = np.roll(np.roll(arr, dx, axis=0), dy, axis=1)
    return (1.0 - alpha) * arr + alpha * shifted

def grid_to_rgb(grid):
    rgba = cm.get_cmap('viridis')(np.clip(grid, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)

# ------------------------ Simulation ------------------------

class MossSim:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        self.step_id = 0
        self.running = False

        self.env = PRESETS[DEFAULT_PRESET].copy()

        # cycles
        self.cycle_on = True
        self.cycle_period = 26
        self.cycle_amp = 0.22

        # spatial params
        self.spatial_on = True
        self.shade_coeff = 0.55
        self.moisture_diff = 0.12
        self.moisture_evap = 0.04
        self.wind_deg = 135.0
        self.wind_mix = 0.05
        self.light_grad = 0.12
        self.sun_deg = 135.0
        self.grad_bias = 1.0
        self.parent_weight = 0.6
        self.spore_rate = 0.02
        self.spore_travel = 10.0

        self.reset_fields(self.env)

    def reset_fields(self, env):
        self.grid[...] = 0.0
        c = GRID_SIZE//2
        self.grid[c, c] = INIT_BIOMASS
        self.M = np.full_like(self.grid, fill_value=env['humidity'], dtype=float)
        self.step_id = 0

    def set_env(self, env):
        self.env = env.copy()

    def coverage(self):
        return float((self.grid >= OCC_THR).mean())

    def mean_biomass(self):
        nz = self.grid[self.grid > 0]
        return 0.0 if nz.size == 0 else float(nz.mean())

    def plant_seed_rc(self, r, c, radius=0):
        """Plant a seed at row r, col c (plus optional tiny neighborhood)."""
        r = int(np.clip(r, 0, GRID_SIZE-1))
        c = int(np.clip(c, 0, GRID_SIZE-1))
        self.grid[r, c] = max(self.grid[r, c], INIT_BIOMASS*1.2)
        if radius > 0:
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    rr = (r+dr) % GRID_SIZE
                    cc = (c+dc) % GRID_SIZE
                    self.grid[rr, cc] = max(self.grid[rr, cc], INIT_BIOMASS*0.6)

    def spore_burst(self):
        if self.spore_rate <= 0 or np.random.rand() > self.spore_rate:
            return
        occ = np.argwhere(self.grid >= OCC_THR)
        if occ.size == 0:
            return
        idx = occ[np.random.randint(len(occ))]
        th = np.deg2rad(self.wind_deg)
        d = max(1.0, np.random.normal(self.spore_travel, self.spore_travel*0.25))
        dx = int(round(d * np.sin(th)))
        dy = int(round(d * np.cos(th)))
        x = (idx[0] + dx) % GRID_SIZE
        y = (idx[1] + dy) % GRID_SIZE
        self.grid[x, y] = max(self.grid[x, y], 0.20*np.random.uniform(0.7, 1.3))

    def step(self):
        self.step_id += 1

        # Update moisture & light
        base_h = self.env['humidity']
        if self.cycle_on:
            base_h = moisture_pulse(self.step_id, base_h, self.cycle_period, self.cycle_amp)

        if self.spatial_on:
            self.M = diffuse(self.M, self.moisture_diff)
            self.M = np.clip((1.0 - self.moisture_evap)*self.M + 0.30*base_h, 0.0, 1.0)
            self.M = advect(self.M, self.wind_deg, self.wind_mix)
            L_field = make_light_field(self.env['light'], self.light_grad, self.sun_deg)
            L_eff = apply_shade(L_field, self.grid, self.shade_coeff)
        else:
            self.M = np.full_like(self.M, fill_value=base_h)
            L_eff = np.full_like(self.grid, fill_value=self.env['light'])

        # Suitability map
        local_env = dict(
            humidity=self.M,
            light=L_eff,
            temperature=self.env['temperature'],
            pH=self.env['pH'],
            nutrients=self.env['nutrients'],
        )
        raw = suitability_raw(local_env)
        S = logistic(raw)
        neigh = neighborhood_mean(self.grid)

        # Colonization probability
        base_col = SPREAD_BASE * S * (1.0 + NEIGHBOR_WEIGHT*neigh)
        if self.spatial_on and self.grad_bias > 0:
            colonize_prob = np.zeros_like(self.grid)
            for (dx,dy) in DIRS:
                parent_occ = np.roll(self.grid >= OCC_THR, shift=(dx,dy), axis=(0,1))
                parent_str = np.roll(self.grid, shift=(dx,dy), axis=(0,1))
                S_src = np.roll(S, shift=(dx,dy), axis=(0,1))
                advantage = np.clip(S - S_src, 0.0, None)
                Pd = base_col * parent_occ * (1.0 + self.grad_bias*advantage) * (1.0 + self.parent_weight*parent_str)
                Pd = np.clip(Pd, 0.0, 1.0)
                colonize_prob = 1.0 - (1.0 - colonize_prob) * (1.0 - Pd)
        else:
            colonize_prob = np.clip(base_col, 0.0, 1.0)

        growth = 0.10 * S * (1.0 + 0.5 * neigh)
        decay  = DECAY_RATE * (1.0 - S)

        new_grid = self.grid.copy()
        empty = self.grid < OCC_THR
        rand = np.random.rand(GRID_SIZE, GRID_SIZE)
        colonize_mask = empty & (rand < colonize_prob)
        if colonize_mask.any():
            new_grid[colonize_mask] = INIT_BIOMASS * (0.8 + 0.4*np.random.rand(np.sum(colonize_mask)))

        occ_mask = self.grid >= OCC_THR
        new_grid[occ_mask] = np.minimum(MAX_BIOMASS, new_grid[occ_mask] + growth[occ_mask])

        poor_mask = S < 0.45
        new_grid[poor_mask] = np.maximum(0.0, new_grid[poor_mask] - decay[poor_mask])

        new_grid += NOISE * (np.random.rand(GRID_SIZE, GRID_SIZE) - 0.5)
        self.grid = np.clip(new_grid, 0.0, MAX_BIOMASS)

        if self.spatial_on and self.spore_rate > 0:
            self.spore_burst()

    def snapshot(self):
        return grid_to_rgb(self.grid)

# ------------------------ Gradio App ------------------------

SIM = MossSim()

def do_start():
    SIM.running = True
    cov, mb = SIM.coverage(), SIM.mean_biomass()
    return f"▶ Running | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def do_pause():
    SIM.running = not SIM.running
    cov, mb = SIM.coverage(), SIM.mean_biomass()
    state = "▶ Running" if SIM.running else "⏸ Paused"
    return f"{state} | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def do_step():
    was = SIM.running
    SIM.running = False
    SIM.step()
    img = SIM.snapshot()
    cov, mb = SIM.coverage(), SIM.mean_biomass()
    SIM.running = was
    return img, f"Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def do_reset(preset_name):
    env = PRESETS[preset_name]
    SIM.reset_fields(env)
    SIM.set_env(env)
    img = SIM.snapshot()
    return img, f"Reset to preset: {preset_name}"

def apply_preset(preset_name):
    env = PRESETS[preset_name]
    SIM.set_env(env)
    cov, mb = SIM.coverage(), SIM.mean_biomass()
    return env['humidity'], env['light'], env['temperature'], env['pH'], env['nutrients'], f"Preset → {preset_name} | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def set_env(h, l, t, pH, n):
    SIM.set_env(dict(humidity=float(h), light=float(l), temperature=float(t), pH=float(pH), nutrients=float(n)))
    cov, mb = SIM.coverage(), SIM.mean_biomass()
    return f"Env updated | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def set_cycles(on, period, amp):
    SIM.cycle_on = bool(on)
    SIM.cycle_period = int(period)
    SIM.cycle_amp = float(amp)
    cov, mb = SIM.coverage(), SIM.mean_biomass()
    return f"Cycles: {'ON' if SIM.cycle_on else 'OFF'} (period={SIM.cycle_period}, amp={SIM.cycle_amp:.2f}) | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def set_spatial(on, shade, diff, evap, wdeg, wmix, lgrad, sdeg, gbias, pwt, srate, stravel):
    SIM.spatial_on = bool(on)
    SIM.shade_coeff   = float(shade)
    SIM.moisture_diff = float(diff)
    SIM.moisture_evap = float(evap)
    SIM.wind_deg      = float(wdeg)
    SIM.wind_mix      = float(wmix)
    SIM.light_grad    = float(lgrad)
    SIM.sun_deg       = float(sdeg)
    SIM.grad_bias     = float(gbias)
    SIM.parent_weight = float(pwt)
    SIM.spore_rate    = float(srate)
    SIM.spore_travel  = float(stravel)
    cov, mb = SIM.coverage(), SIM.mean_biomass()
    return f"Spatial: {'ON' if SIM.spatial_on else 'OFF'} | Shade={SIM.shade_coeff:.2f}, Diff={SIM.moisture_diff:.2f}, Evap={SIM.moisture_evap:.2f}, Wind={SIM.wind_deg:.0f}°/{SIM.wind_mix:.2f}, LightGrad={SIM.light_grad:.2f}@{SIM.sun_deg:.0f}°, GradBias={SIM.grad_bias:.2f}, SporeRate={SIM.spore_rate:.2f}, Travel≈{SIM.spore_travel:.0f} | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def plant_on_click(evt: gr.SelectData):
    """
    Click-to-seed handler. We assume evt.index returns (x, y) pixel coords
    relative to the underlying numpy image (which we keep at GRID_SIZE x GRID_SIZE).
    """
    try:
        x, y = evt.index  # (col, row)
    except Exception:
        return SIM.snapshot(), "Click failed (no index)."

    # Map to grid (clamp and swap to row,col)
    c = int(np.clip(round(x), 0, GRID_SIZE-1))
    r = int(np.clip(round(y), 0, GRID_SIZE-1))
    SIM.plant_seed_rc(r, c, radius=1)
    img = SIM.snapshot()
    cov, mb = SIM.coverage(), SIM.mean_biomass()
    return img, f"Planted at (r={r}, c={c}) | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def tick(steps_per_tick:int):
    steps = int(max(1, steps_per_tick))
    if SIM.running:
        for _ in range(steps):
            SIM.step()
    img = SIM.snapshot()
    cov, mb = SIM.coverage(), SIM.mean_biomass()
    e = SIM.env
    status = (f"{'▶ Running' if SIM.running else '⏸ Paused'} | Step {SIM.step_id} | "
              f"Coverage {cov:.2f} | Mean biomass {mb:.2f} | "
              f"H={e['humidity']:.2f}{'~' if SIM.cycle_on else ''}, L≈{e['light']:.2f}, "
              f"T={e['temperature']:.1f}°C, pH={e['pH']:.1f}, N={e['nutrients']:.2f} | "
              f"Speed={steps} step(s)/tick")
    return img, status

# Build UI
with gr.Blocks(title="Moss Seed — Spatial + Click + Speed") as demo:
    gr.Markdown("# 🌱 Moss Seed — Spatially Aware (Click-to-Seed + Speed Control)")
    gr.Markdown("**Click on the image to plant a new seed.** Use the speed slider to accelerate the simulation "
                "(it performs multiple steps each tick). Toggle spatial awareness for microclimate & gradient following.")

    with gr.Row():
        canvas = gr.Image(label="Moss Biomass (0–1)", image_mode="RGB", interactive=True, height=520)

    with gr.Row():
        start_btn = gr.Button("Start", variant="primary")
        pause_btn = gr.Button("Pause/Resume")
        step_btn  = gr.Button("Step")
        reset_btn = gr.Button("Reset to Preset")

    with gr.Row():
        preset = gr.Dropdown(choices=list(PRESETS.keys()), value=DEFAULT_PRESET, label="Environment Preset")

    # Environment sliders
    env = PRESETS[DEFAULT_PRESET]
    with gr.Row():
        humidity = gr.Slider(0, 1, value=env['humidity'], step=0.01, label="Humidity")
        light    = gr.Slider(0, 1, value=env['light'], step=0.01, label="Light")
        temp     = gr.Slider(-10, 40, value=env['temperature'], step=0.5, label="Temperature (°C)")
        ph       = gr.Slider(3.5, 9.0, value=env['pH'], step=0.1, label="pH")
        nutr     = gr.Slider(0, 1, value=env['nutrients'], step=0.01, label="Nutrients")

    # Cycles
    with gr.Row():
        cyc_on  = gr.Checkbox(value=True, label="Wet–Dry Cycles ON")
        cyc_per = gr.Slider(5, 120, value=26, step=1, label="Cycle Period (steps)")
        cyc_amp = gr.Slider(0.0, 0.5, value=0.22, step=0.01, label="Cycle Amplitude")

    # Spatial awareness controls
    with gr.Accordion("Spatial Awareness & Fields", open=True):
        with gr.Row():
            spatial_on = gr.Checkbox(value=True, label="Spatial Awareness ON")
            shade      = gr.Slider(0.0, 1.0, value=0.55, step=0.01, label="Shade Coeff (light blocked by biomass)")
            diff       = gr.Slider(0.0, 0.5, value=0.12, step=0.01, label="Moisture Diffusion")
            evap       = gr.Slider(0.0, 0.2, value=0.04, step=0.005, label="Moisture Evaporation")
        with gr.Row():
            wind_deg   = gr.Slider(0, 360, value=135, step=1, label="Wind Direction (°)")
            wind_mix   = gr.Slider(0.0, 0.3, value=0.05, step=0.005, label="Wind Drift (moisture advection)")
            lgrad      = gr.Slider(0.0, 0.5, value=0.12, step=0.01, label="Light Gradient Strength")
            sun_deg    = gr.Slider(0, 360, value=135, step=1, label="Sun Direction (°)")
        with gr.Row():
            gbias      = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="Gradient Bias (climb suitability)")
            pwt        = gr.Slider(0.0, 2.0, value=0.6, step=0.05, label="Parent Strength Weight")
            srate      = gr.Slider(0.0, 0.1, value=0.02, step=0.005, label="Spore Burst Rate (per step)")
            stravel    = gr.Slider(1.0, 40.0, value=10.0, step=1.0, label="Spore Travel (cells)")

    with gr.Row():
        speed = gr.Slider(1, 50, value=6, step=1, label="Speed — Steps per Tick (↑ faster)")

    status = gr.Textbox(label="Status / Telemetry", value="Ready.", interactive=False)

    # Initial render
    canvas.value = SIM.snapshot()

    # Wire events
    start_btn.click(fn=do_start, outputs=status)
    pause_btn.click(fn=do_pause, outputs=status)
    step_btn.click(fn=do_step, outputs=[canvas, status])
    reset_btn.click(fn=do_reset, inputs=preset, outputs=[canvas, status])
    preset.change(fn=apply_preset, inputs=preset, outputs=[humidity, light, temp, ph, nutr, status])

    for ctrl in (humidity, light, temp, ph, nutr):
        ctrl.change(fn=set_env, inputs=[humidity, light, temp, ph, nutr], outputs=status)

    cyc_on.change(fn=set_cycles, inputs=[cyc_on, cyc_per, cyc_amp], outputs=status)
    cyc_per.change(fn=set_cycles, inputs=[cyc_on, cyc_per, cyc_amp], outputs=status)
    cyc_amp.change(fn=set_cycles, inputs=[cyc_on, cyc_per, cyc_amp], outputs=status)

    spatial_inputs = [spatial_on, shade, diff, evap, wind_deg, wind_mix, lgrad, sun_deg, gbias, pwt, srate, stravel]
    for ctrl in spatial_inputs:
        ctrl.change(fn=set_spatial, inputs=spatial_inputs, outputs=status)

    # Click-to-seed
    canvas.select(fn=plant_on_click, outputs=[canvas, status])

    # Timer for live updates; pass speed as input so tick can run multiple steps
    demo.load(fn=lambda s: tick(s), inputs=speed, outputs=[canvas, status])
    gr.Timer(0.12).tick(fn=lambda s: tick(s), inputs=speed, outputs=[canvas, status])

if __name__ == "__main__":
    demo.launch()
