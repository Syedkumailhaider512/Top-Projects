
"""
Moss Seed — Evolution + Memory Brain (v0.9, Smooth Playback)
Author: Nova (for Kumail / AI Cadmey)

New in v0.9
- ✅ **Evolution power**:
    • Phenotypic plasticity (online adjustment of optima toward encountered microclimate).
    • Heritable mutation on colonization events (new genotypes arise from parents).
- ✅ **Memory Brain**:
    • Tracks each genotype's optima/tolerances, ancestry, exposures, biomass (fitness proxy).
    • Save / Load brain (.json) to persist adaptability patterns across runs.
- ✅ **Smooth playback**: 1 step per frame (no jumps). FPS slider controls visual speed.
- ✅ **Click-to-seed** anywhere.

Run:
  pip install gradio numpy matplotlib
  python moss_seed_evolution_brain_v0_9.py
"""

import json, os, math, time, random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

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
OCC_THR = 0.05

# Ranges to clamp evolved parameters
R = dict(humidity=(0.0,1.0), light=(0.0,1.0), temperature=(-10.0,40.0), pH=(3.5,9.0), nutrients=(0.0,1.0))

# Default starting genotype
DEFAULT_OPTIMA = dict(humidity=0.75, light=0.45, temperature=12.0, pH=5.5, nutrients=0.35)
DEFAULT_TOL    = dict(humidity=0.25, light=0.35, temperature=10.0, pH=1.5, nutrients=0.30)

PRESETS = {
    "Urban Wall (shaded, intermittent wet)": dict(humidity=0.55, light=0.35, temperature=14.0, pH=7.2, nutrients=0.30),
    "Temperate Forest Floor":                dict(humidity=0.80, light=0.30, temperature=10.0, pH=5.5, nutrients=0.35),
    "Boreal/Arctic Tundra":                  dict(humidity=0.60, light=0.55, temperature=2.0,  pH=5.0, nutrients=0.25),
    "Desert Rock (ephemeral rain)":          dict(humidity=0.25, light=0.80, temperature=28.0, pH=7.8, nutrients=0.15),
    "Rainforest Trunk (humid & shaded)":     dict(humidity=0.95, light=0.25, temperature=22.0, pH=5.2, nutrients=0.45),
}
DEFAULT_PRESET = "Urban Wall (shaded, intermittent wet)"

DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# ------------------------ Utilities ------------------------

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def clamp(v, lo, hi):
    return float(min(max(v, lo), hi))

def clamp_dict(d, ranges):
    out = {}
    for k,v in d.items():
        lo,hi = ranges[k]
        out[k] = clamp(float(v), lo, hi)
    return out

def suitability_raw(env, optima, tol):
    # support scalar or array env values
    d_h = abs(env['humidity']    - optima['humidity'])    / (tol['humidity'] + 1e-6)
    d_l = abs(env['light']       - optima['light'])       / (tol['light'] + 1e-6)
    d_t = abs(env['temperature'] - optima['temperature']) / (tol['temperature'] + 1e-6)
    d_p = abs(env['pH']          - optima['pH'])          / (tol['pH'] + 1e-6)
    d_n = abs(env['nutrients']   - optima['nutrients'])   / (tol['nutrients'] + 1e-6)
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
    return np.clip(light_field * (1.0 - shade_coeff*shade), 0.0, 1.0)

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
    dx = int(round(np.sin(th)))   # y axis downward
    dy = int(round(np.cos(th)))
    shifted = np.roll(np.roll(arr, dx, axis=0), dy, axis=1)
    return (1.0 - alpha) * arr + alpha * shifted

def grid_to_rgb(grid, gid, palette):
    # biomass brightness + hue by genotype
    # palette: list of RGB colors (0..255)
    h, w = grid.shape
    img = np.zeros((h,w,3), dtype=np.uint8)
    # brightness from biomass
    b = np.clip(grid, 0.0, 1.0)
    # color per gid
    for g in range(len(palette)):
        mask = (gid == g) & (grid >= OCC_THR)
        if not np.any(mask):
            continue
        color = np.array(palette[g], dtype=float)
        # scale color by biomass brightness (simple multiply toward white)
        img[mask] = np.clip((b[mask][:,None] * color), 0, 255).astype(np.uint8)
    # faint base for sub-threshold biomass
    faint = (grid > 0) & (grid < OCC_THR)
    img[faint] = np.clip((grid[faint][:,None] * np.array([160,160,160])),0,255).astype(np.uint8)
    return img

def default_palette(n=16, seed=7):
    rng = np.random.default_rng(seed)
    cols = []
    for i in range(n):
        c = rng.integers(50, 255, size=3)
        cols.append(tuple(int(x) for x in c))
    return cols

# ------------------------ Genotype & Brain ------------------------

@dataclass
class Genotype:
    gid: int
    optima: Dict[str, float]
    tol: Dict[str, float]
    parent: Optional[int] = None
    birth_step: int = 0
    mut_history: List[Dict] = field(default_factory=list)
    exposure: Dict[str, float] = field(default_factory=lambda: {"count":0.0, "humidity":0.0, "light":0.0, "temperature":0.0, "pH":0.0, "nutrients":0.0, "stress_sum":0.0})
    biomass_sum: float = 0.0  # updated each frame

    def to_dict(self):
        return {
            "gid": self.gid,
            "optima": self.optima,
            "tol": self.tol,
            "parent": self.parent,
            "birth_step": self.birth_step,
            "mut_history": self.mut_history,
            "exposure": self.exposure,
            "biomass_sum": self.biomass_sum
        }

# ------------------------ Simulation ------------------------

class MossSim:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        self.gid  = -np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int16)  # -1 empty
        self.step_id = 0
        self.running = False

        self.env = PRESETS[DEFAULT_PRESET].copy()

        # cycles
        self.cycle_on = True
        self.cycle_period = 26
        self.cycle_amp = 0.22

        # spatial
        self.spatial_on = True
        self.shade_coeff = 0.55
        self.moisture_diff = 0.12
        self.moisture_evap = 0.04
        self.wind_deg = 135.0
        self.wind_mix = 0.05
        self.light_grad = 0.12
        self.sun_deg = 135.0

        # evolution
        self.evolve_on = True
        self.plasticity_rate = 0.02     # optima learning rate
        self.mutation_rate   = 0.02     # chance per colonization to mutate
        self.mutation_scale  = 0.15     # fraction of tolerance for step size
        self.max_genotypes   = 12

        # spore dispersal
        self.grad_bias = 1.0
        self.parent_weight = 0.6
        self.spore_rate = 0.015
        self.spore_travel = 10.0

        # fps control
        self.BASE_FPS = 33.0
        self.skip_n = 2
        self._tick_count = 0

        # Brain / genotypes
        self.palette = default_palette(self.max_genotypes)
        self.genotypes : List[Genotype]= []
        self._next_gid = 0

        self.reset_fields(self.env)

    # ----- Brain helpers -----
    def new_genotype(self, optima, tol, parent=None):
        g = Genotype(gid=self._next_gid, optima=optima.copy(), tol=tol.copy(), parent=parent, birth_step=self.step_id)
        self.genotypes.append(g)
        self._next_gid += 1
        return g.gid

    def seed_center(self):
        c = GRID_SIZE//2
        self.grid[c,c] = max(self.grid[c,c], INIT_BIOMASS)
        if not self.genotypes:
            gid0 = self.new_genotype(DEFAULT_OPTIMA, DEFAULT_TOL, parent=None)
        else:
            gid0 = 0
        self.gid[c,c] = gid0

    def reset_fields(self, env):
        self.grid[...] = 0.0
        self.gid[...]  = -1
        self.M = np.full((GRID_SIZE, GRID_SIZE), fill_value=env['humidity'], dtype=float)
        self.step_id = 0
        self._tick_count = 0
        self.genotypes = []
        self._next_gid = 0
        self.seed_center()

    # ----- Public controls -----
    def set_env(self, env): self.env = env.copy()
    def set_fps(self, fps):
        fps = float(np.clip(fps, 2.0, self.BASE_FPS))
        k = int(round(self.BASE_FPS / max(1.0, fps)))
        self.skip_n = max(1, k)

    def plant_seed_rc(self, r, c, radius=1, gid_hint=None):
        r = int(np.clip(r, 0, GRID_SIZE-1)); c = int(np.clip(c, 0, GRID_SIZE-1))
        if gid_hint is None:
            gid_hint = 0 if self.genotypes else self.new_genotype(DEFAULT_OPTIMA, DEFAULT_TOL, parent=None)
        self.grid[r,c] = max(self.grid[r,c], INIT_BIOMASS*1.2)
        self.gid[r,c]  = gid_hint
        if radius>0:
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    rr = (r+dr) % GRID_SIZE; cc = (c+dc) % GRID_SIZE
                    self.grid[rr,cc] = max(self.grid[rr,cc], INIT_BIOMASS*0.6)
                    self.gid[rr,cc]  = gid_hint

    def coverage(self): return float((self.grid >= OCC_THR).mean())
    def mean_biomass(self):
        nz = self.grid[self.grid>0]; return 0.0 if nz.size==0 else float(nz.mean())

    # ----- Evolution mechanics -----
    def mutate_child(self, parent_gid):
        parent = self.genotypes[parent_gid]
        new_opt = {}
        new_tol = {}
        for k in parent.optima:
            step = self.mutation_scale * parent.tol[k]
            new_opt[k] = clamp(np.random.normal(parent.optima[k], step), *R[k])
            # allow slight change in tolerances too (stay positive)
            ts = 0.25 * parent.tol[k]
            new_tol[k] = max(0.05, np.random.normal(parent.tol[k], ts))
        new_gid = self.new_genotype(new_opt, new_tol, parent=parent_gid)
        # record
        self.genotypes[new_gid].mut_history.append({"from": parent_gid, "step": self.step_id})
        return new_gid

    def plasticity_update(self, gid_mask, optima, rate, env_maps, S):
        """Move optima slightly toward experienced env proportional to stress.
        Robust to scalar env maps (temperature/pH/nutrients) or per-cell arrays (humidity/light).
        """
        if not np.any(gid_mask):
            return optima
        # stress averaged on the occupied genotype cells
        if np.ndim(S) == 0:
            stress = (1.0 - float(S))
        else:
            stress = (1.0 - S)[gid_mask].mean() if np.any(gid_mask) else 0.0
        for k in optima.keys():
            env_k = env_maps[k]
            if np.ndim(env_k) == 0:
                env_mean = float(env_k)
            else:
                env_mean = float(env_k[gid_mask].mean()) if np.any(gid_mask) else float(env_k.mean())
            optima[k] = clamp(optima[k] + rate * stress * (env_mean - optima[k]), *R[k])
        return optima

    # ----- Step -----
    def step(self):
        self.step_id += 1

        # Update fields
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

        # Local environment maps
        env_maps = dict(humidity=self.M, light=L_eff, temperature=self.env['temperature'],
                        pH=self.env['pH'], nutrients=self.env['nutrients'])

        # Suitability per cell depends on cell's genotype (if occupied), else use a neutral baseline
        S_occ = np.zeros_like(self.grid)
        for g in self.genotypes:
            mask = (self.gid == g.gid) & (self.grid > 0)
            if not np.any(mask):
                g.biomass_sum = 0.0
                continue
            raw = suitability_raw(env_maps, g.optima, g.tol)
            # if raw is scalar, broadcast; else per-cell
            Sg = logistic(raw if np.ndim(raw)>0 else np.full_like(self.grid, raw))
            S_occ[mask] = Sg[mask]
            # Update fitness proxy
            g.biomass_sum = float(self.grid[mask].sum())

        # Neighborhood mean biomass for growth + colonization
        neigh_b = neighborhood_mean(self.grid)

        # Base colonization prob field (we'll enrich with gradient + parent strength per genotype flows)
        base_col = SPREAD_BASE * np.clip(S_occ + 0.2, 0.0, 1.0) * (1.0 + NEIGHBOR_WEIGHT*neigh_b)

        # Build colonization prob by combining directed contributions; also record best source dir for assignment
        colonize_prob = np.zeros_like(self.grid)
        best_dir_idx = -np.ones_like(self.gid, dtype=np.int8)

        for di,(dx,dy) in enumerate(DIRS):
            parent_occ = np.roll(self.grid >= OCC_THR, shift=(dx,dy), axis=(0,1))
            parent_str = np.roll(self.grid, shift=(dx,dy), axis=(0,1))
            gid_src    = np.roll(self.gid, shift=(dx,dy), axis=(0,1))
            S_src      = np.roll(S_occ, shift=(dx,dy), axis=(0,1))

            advantage = np.clip(S_occ - S_src, 0.0, None)
            Pd = base_col * parent_occ * (1.0 + self.grad_bias*advantage) * (1.0 + self.parent_weight*parent_str)
            Pd = np.clip(Pd, 0.0, 1.0)

            # union combine
            new_union = 1.0 - (1.0 - colonize_prob) * (1.0 - Pd)
            # choose best dir by Pd where it increases union
            better = new_union > colonize_prob
            best_dir_idx[better] = di
            colonize_prob = new_union

        # Growth & decay
        growth = 0.10 * np.clip(S_occ, 0.0, 1.0) * (1.0 + 0.5*neigh_b)
        decay  = DECAY_RATE * (1.0 - np.clip(S_occ, 0.0, 1.0))

        new_grid = self.grid.copy()
        new_gid  = self.gid.copy()

        # Colonization events
        empty = self.grid < OCC_THR
        rand = np.random.rand(GRID_SIZE, GRID_SIZE)
        col_mask = empty & (rand < colonize_prob)

        if np.any(col_mask):
            # assign genotype from best source direction (rolled gid)
            src_idx = best_dir_idx[col_mask]
            # initialize with parent gid inferred from dir
            # We'll vectorize by constructing arrays per dir and picking with boolean masks
            parent_gid_field = -np.ones_like(self.gid)
            for di,(dx,dy) in enumerate(DIRS):
                pg = np.roll(self.gid, shift=(dx,dy), axis=(0,1))
                parent_gid_field[best_dir_idx==di] = pg[best_dir_idx==di]
            # Now for new cells:
            parents = parent_gid_field[col_mask].astype(int)
            # mutation decision per event
            mutate_flags = (np.random.rand(parents.size) < self.mutation_rate) if self.evolve_on else np.zeros(parents.size, dtype=bool)

            # Ensure capacity for new mutants (cap by max_genotypes)
            for i,(pr,mut) in enumerate(zip(parents, mutate_flags)):
                gid_assigned = pr
                if mut:
                    # if capacity not full, spawn; else mutate parent in-place (microevolution)
                    if len(self.genotypes) < self.max_genotypes and pr >= 0:
                        gid_assigned = self.mutate_child(pr)
                    elif pr >= 0:
                        # microevolve parent optima slightly
                        p = self.genotypes[pr]
                        for k in p.optima.keys():
                            step = 0.5*self.mutation_scale * p.tol[k]
                            p.optima[k] = clamp(np.random.normal(p.optima[k], step), *R[k])
                        gid_assigned = pr
                # write gid to corresponding index in new_gid where col_mask true
                # We have to map i-th True index to positions
                # Simpler: get indices of col_mask and assign
                # We'll defer assignment after loop
                mutate_flags[i] = (gid_assigned != pr)  # not used but okay
            # Assign gids vectorized
            rr, cc = np.where(col_mask)
            # recompute parents (since we may have created new gids, but assignment is unaffected)
            parent_gid_field = -np.ones_like(self.gid)
            for di,(dx,dy) in enumerate(DIRS):
                pg = np.roll(self.gid, shift=(dx,dy), axis=(0,1))
                parent_gid_field[best_dir_idx==di] = pg[best_dir_idx==di]
            parents = parent_gid_field[col_mask].astype(int)

            # For a fraction mutate to new gids (if evolve_on & capacity allows)
            if self.evolve_on and len(self.genotypes) < self.max_genotypes:
                # choose a subset to mutate
                mut_choose = (np.random.rand(len(rr)) < self.mutation_rate)
                # generate actual new gid for each True
                for idx,j in enumerate(np.where(mut_choose)[0]):
                    pr = parents[j]
                    if pr >= 0:
                        parents[j] = self.mutate_child(pr)

            new_grid[rr, cc] = INIT_BIOMASS * (0.8 + 0.4*np.random.rand(len(rr)))
            new_gid[rr, cc]  = parents

        # Grow occupied cells
        occ_mask = self.grid >= OCC_THR
        new_grid[occ_mask] = np.minimum(MAX_BIOMASS, new_grid[occ_mask] + growth[occ_mask])

        # Decompose in poor suitability
        poor_mask = S_occ < 0.45
        new_grid[poor_mask] = np.maximum(0.0, new_grid[poor_mask] - decay[poor_mask])
        # If biomass becomes ~0, clear gid
        died = new_grid < 1e-4
        new_gid[died] = -1

        # noise
        new_grid += NOISE * (np.random.rand(GRID_SIZE, GRID_SIZE) - 0.5)
        self.grid = np.clip(new_grid, 0.0, MAX_BIOMASS)
        self.gid  = new_gid

        # Spore long-range
        if self.spatial_on and self.spore_rate > 0 and np.random.rand() < self.spore_rate:
            occ = np.argwhere(self.grid >= OCC_THR)
            if occ.size > 0:
                idx = occ[np.random.randint(len(occ))]
                th = np.deg2rad(self.wind_deg)
                d = max(1.0, np.random.normal(self.spore_travel, self.spore_travel*0.25))
                dx = int(round(d * np.sin(th))); dy = int(round(d * np.cos(th)))
                x = (idx[0] + dx) % GRID_SIZE; y = (idx[1] + dy) % GRID_SIZE
                self.grid[x,y] = max(self.grid[x,y], 0.20*np.random.uniform(0.7, 1.3))
                self.gid[x,y]  = self.gid[idx[0], idx[1]]

        # ---- Plasticity + Brain exposure stats ----
        if self.evolve_on:
            # Build S map per genotype and update optima toward experienced env with stress factor
            for g in self.genotypes:
                mask = (self.gid == g.gid) & (self.grid > 0)
                if not np.any(mask):
                    continue
                # compute S for this genotype at its cells
                raw = suitability_raw(env_maps, g.optima, g.tol)
                Sg = logistic(raw if np.ndim(raw)>0 else np.full_like(self.grid, raw))
                # plasticity update
                g.optima = self.plasticity_update(mask, g.optima, self.plasticity_rate, env_maps, Sg)
                # exposure stats (running mean)
                cnt = mask.sum()
                g.exposure["count"] += float(cnt)
                for k in ["humidity","light","temperature","pH","nutrients"]:
                    g.exposure[k] += float(env_maps[k][mask].mean()) if np.ndim(env_maps[k])>0 else float(env_maps[k])
                g.exposure["stress_sum"] += float((1.0 - Sg[mask]).mean() if np.any(mask) else 0.0)

    # ----- Snapshot & Brain -----
    def snapshot(self):
        # ensure palette size
        if len(self.palette) < max(1,len(self.genotypes)):
            self.palette = default_palette(max(16, len(self.genotypes)+4))
        return grid_to_rgb(self.grid, self.gid, self.palette)

    def brain_summary_text(self, top_k=3):
        if not self.genotypes:
            return "No genotypes."
        # sort by biomass_sum desc
        gs = sorted(self.genotypes, key=lambda g: g.biomass_sum, reverse=True)[:top_k]
        lines = []
        for g in gs:
            lines.append(f"G{g.gid} (parent={g.parent}, born@{g.birth_step}) biomass≈{g.biomass_sum:.1f}")
            lines.append(f"  optima: H={g.optima['humidity']:.2f}, L={g.optima['light']:.2f}, T={g.optima['temperature']:.1f}, pH={g.optima['pH']:.1f}, N={g.optima['nutrients']:.2f}")
            lines.append(f"  tol:    H={g.tol['humidity']:.2f}, L={g.tol['light']:.2f}, T={g.tol['temperature']:.1f}, pH={g.tol['pH']:.1f}, N={g.tol['nutrients']:.2f}")
        return "\n".join(lines)

    def save_brain(self, path):
        brain = {
            "version": "moss_brain_v1",
            "step": self.step_id,
            "genotypes": [g.to_dict() for g in self.genotypes],
            "env": self.env,
            "notes": "Stores adaptive optima/tolerances and exposure summaries per genotype."
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(brain, f, indent=2)
        return path

    def load_brain(self, path):
        with open(path, "r", encoding="utf-8") as f:
            brain = json.load(f)
        self.genotypes = []
        self._next_gid = 0
        for g in brain.get("genotypes", []):
            gid = self.new_genotype(g["optima"], g["tol"], parent=g.get("parent"))
            self.genotypes[gid].birth_step = g.get("birth_step", 0)
            self.genotypes[gid].mut_history = g.get("mut_history", [])
            self.genotypes[gid].exposure = g.get("exposure", self.genotypes[gid].exposure)
        # keep current grid; optionally reseed center with dominant genotype 0
        return len(self.genotypes)

# ------------------------ Gradio App ------------------------

SIM = MossSim()

def do_start(): SIM.running=True; cov,mb=SIM.coverage(), SIM.mean_biomass(); return f"▶ Running | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"
def do_pause(): SIM.running=not SIM.running; cov,mb=SIM.coverage(), SIM.mean_biomass(); return f"{'▶ Running' if SIM.running else '⏸ Paused'} | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"
def do_step():
    was=SIM.running; SIM.running=False; SIM.step(); img=SIM.snapshot(); cov,mb=SIM.coverage(),SIM.mean_biomass(); SIM.running=was
    return img, f"Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def do_reset(preset_name):
    env = PRESETS[preset_name]; SIM.reset_fields(env); SIM.set_env(env); img=SIM.snapshot()
    return img, f"Reset to preset: {preset_name}"

def apply_preset(preset_name):
    env = PRESETS[preset_name]; SIM.set_env(env); cov,mb=SIM.coverage(),SIM.mean_biomass()
    return env['humidity'], env['light'], env['temperature'], env['pH'], env['nutrients'], f"Preset → {preset_name} | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def set_env(h,l,t,pH,n):
    SIM.set_env(dict(humidity=float(h), light=float(l), temperature=float(t), pH=float(pH), nutrients=float(n)))
    cov,mb=SIM.coverage(),SIM.mean_biomass()
    return f"Env updated | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def set_cycles(on,period,amp):
    SIM.cycle_on=bool(on); SIM.cycle_period=int(period); SIM.cycle_amp=float(amp); cov,mb=SIM.coverage(),SIM.mean_biomass()
    return f"Cycles: {'ON' if SIM.cycle_on else 'OFF'} (period={SIM.cycle_period}, amp={SIM.cycle_amp:.2f}) | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def set_spatial(on,shade,diff,evap,wdeg,wmix,lgrad,sdeg):
    SIM.spatial_on=bool(on); SIM.shade_coeff=float(shade); SIM.moisture_diff=float(diff); SIM.moisture_evap=float(evap)
    SIM.wind_deg=float(wdeg); SIM.wind_mix=float(wmix); SIM.light_grad=float(lgrad); SIM.sun_deg=float(sdeg)
    cov,mb=SIM.coverage(),SIM.mean_biomass()
    return f"Spatial: {'ON' if SIM.spatial_on else 'OFF'} | Shade={SIM.shade_coeff:.2f}, Diff={SIM.moisture_diff:.2f}, Evap={SIM.moisture_evap:.2f}, Wind={SIM.wind_deg:.0f}°/{SIM.wind_mix:.2f}, LightGrad={SIM.light_grad:.2f}@{SIM.sun_deg:.0f}° | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def set_evolution(on, plasticity, mut_rate, mut_scale, max_g):
    SIM.evolve_on=bool(on); SIM.plasticity_rate=float(plasticity); SIM.mutation_rate=float(mut_rate); SIM.mutation_scale=float(mut_scale)
    SIM.max_genotypes=int(max_g);
    if len(SIM.palette) < SIM.max_genotypes: SIM.palette = default_palette(SIM.max_genotypes)
    cov,mb=SIM.coverage(),SIM.mean_biomass()
    return f"Evolution: {'ON' if SIM.evolve_on else 'OFF'} | Plasticity={SIM.plasticity_rate:.3f}, Mutation p={SIM.mutation_rate:.3f}, Scale={SIM.mutation_scale:.2f}, MaxGenos={SIM.max_genotypes} | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def set_playback_fps(fps):
    SIM.set_fps(fps); cov,mb=SIM.coverage(),SIM.mean_biomass()
    return f"Playback FPS ≈ {float(fps):.0f} (1 step/frame) | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def plant_on_click(evt: gr.SelectData):
    try:
        x,y = evt.index
    except Exception:
        return SIM.snapshot(), "Click failed (no index)."
    c = int(np.clip(round(x), 0, GRID_SIZE-1)); r = int(np.clip(round(y), 0, GRID_SIZE-1))
    # Plant with dominant genotype if exists else 0
    dom = 0 if not SIM.genotypes else max(SIM.genotypes, key=lambda g: g.biomass_sum).gid
    SIM.plant_seed_rc(r,c,radius=1,gid_hint=dom)
    img = SIM.snapshot(); cov,mb=SIM.coverage(),SIM.mean_biomass()
    return img, f"Planted at (r={r}, c={c}) with G{dom} | Step {SIM.step_id} | Coverage {cov:.2f} | Mean biomass {mb:.2f}"

def tick():
    SIM._tick_count += 1
    if SIM.running and (SIM._tick_count % SIM.skip_n == 0):
        SIM.step()
    img = SIM.snapshot()
    cov, mb = SIM.coverage(), SIM.mean_biomass()
    e = SIM.env
    status = (f"{'▶ Running' if SIM.running else '⏸ Paused'} | Step {SIM.step_id} | "
              f"Coverage {cov:.2f} | Mean biomass {mb:.2f} | "
              f"H={e['humidity']:.2f}{'~' if SIM.cycle_on else ''}, L≈{e['light']:.2f}, "
              f"T={e['temperature']:.1f}°C, pH={e['pH']:.1f}, N={e['nutrients']:.2f}")
    brain_txt = SIM.brain_summary_text(top_k=4)
    return img, status, brain_txt

def save_brain():
    path = "moss_brain.json"
    full = os.path.join(".", path)
    saved = SIM.save_brain(full)
    return saved

def load_brain(file_obj):
    if file_obj is None:
        return "No file."
    n = SIM.load_brain(file_obj.name)
    return f"Loaded {n} genotypes from brain."

# Build UI
with gr.Blocks(title="Moss Seed — Evolution + Memory Brain (Smooth)") as demo:
    gr.Markdown("# 🌱 Moss Seed — Evolution + Memory Brain")
    gr.Markdown("Moss evolves via **plasticity** (optima shift toward experienced microclimate) and **mutation** "
                "on colonization (new genotypes). A **Memory Brain** tracks genotypes’ optima/tolerances and exposures.\n"
                "**One step per frame** → raise FPS to speed visuals without skipping growth stages. Click canvas to plant seeds.")

    with gr.Row():
        canvas = gr.Image(label="Moss Biomass (genotypes colored)", image_mode="RGB", interactive=True, height=520)

    with gr.Row():
        start_btn = gr.Button("Start", variant="primary")
        pause_btn = gr.Button("Pause/Resume")
        step_btn  = gr.Button("Step")
        reset_btn = gr.Button("Reset to Preset")

    with gr.Row():
        preset = gr.Dropdown(choices=list(PRESETS.keys()), value=DEFAULT_PRESET, label="Environment Preset")

    env = PRESETS[DEFAULT_PRESET]
    with gr.Row():
        humidity = gr.Slider(0, 1, value=env['humidity'], step=0.01, label="Humidity")
        light    = gr.Slider(0, 1, value=env['light'], step=0.01, label="Light")
        temp     = gr.Slider(-10, 40, value=env['temperature'], step=0.5, label="Temperature (°C)")
        ph       = gr.Slider(3.5, 9.0, value=env['pH'], step=0.1, label="pH")
        nutr     = gr.Slider(0, 1, value=env['nutrients'], step=0.01, label="Nutrients")

    with gr.Row():
        cyc_on  = gr.Checkbox(value=True, label="Wet–Dry Cycles ON")
        cyc_per = gr.Slider(5, 120, value=26, step=1, label="Cycle Period (steps)")
        cyc_amp = gr.Slider(0.0, 0.5, value=0.22, step=0.01, label="Cycle Amplitude")

    with gr.Accordion("Spatial Fields", open=False):
        with gr.Row():
            spatial_on = gr.Checkbox(value=True, label="Spatial Awareness ON")
            shade      = gr.Slider(0.0, 1.0, value=0.55, step=0.01, label="Shade Coeff")
            diff       = gr.Slider(0.0, 0.5, value=0.12, step=0.01, label="Moisture Diffusion")
            evap       = gr.Slider(0.0, 0.2, value=0.04, step=0.005, label="Moisture Evaporation")
        with gr.Row():
            wind_deg   = gr.Slider(0, 360, value=135, step=1, label="Wind Direction (°)")
            wind_mix   = gr.Slider(0.0, 0.3, value=0.05, step=0.005, label="Wind Drift")
            lgrad      = gr.Slider(0.0, 0.5, value=0.12, step=0.01, label="Light Gradient Strength")
            sun_deg    = gr.Slider(0, 360, value=135, step=1, label="Sun Direction (°)")

    with gr.Accordion("Evolution Settings", open=True):
        with gr.Row():
            evol_on    = gr.Checkbox(value=True, label="Evolution ON")
            plasticity = gr.Slider(0.0, 0.2, value=0.02, step=0.005, label="Plasticity Rate (optima learning)")
            mut_rate   = gr.Slider(0.0, 0.2, value=0.02, step=0.005, label="Mutation Rate (per colonization)")
            mut_scale  = gr.Slider(0.0, 0.5, value=0.15, step=0.01, label="Mutation Step Scale (× tol)")
            max_g      = gr.Slider(1, 24, value=12, step=1, label="Max Genotypes")

    with gr.Row():
        fps = gr.Slider(2, 33, value=16, step=1, label="Playback Speed — FPS (1 step/frame)")

    with gr.Row():
        status = gr.Textbox(label="Status / Telemetry", value="Ready.", interactive=False)

    with gr.Row():
        brain_box = gr.Textbox(label="Memory Brain — Top Genotypes", value="No genotypes yet.", lines=12, interactive=False)


    with gr.Row():
        save_btn = gr.Button("💾 Save Brain (.json)")
        load_in  = gr.File(label="Load Brain (.json)", file_types=[".json"])
        save_out = gr.File(label="Download Brain")

    # Initial image
    canvas.value = SIM.snapshot()
    brain_box.value = SIM.brain_summary_text()

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

    spatial_on.change(fn=set_spatial, inputs=[spatial_on, shade, diff, evap, wind_deg, wind_mix, lgrad, sun_deg], outputs=status)
    for ctrl in (shade, diff, evap, wind_deg, wind_mix, lgrad, sun_deg):
        ctrl.change(fn=set_spatial, inputs=[spatial_on, shade, diff, evap, wind_deg, wind_mix, lgrad, sun_deg], outputs=status)

    evol_on.change(fn=set_evolution, inputs=[evol_on, plasticity, mut_rate, mut_scale, max_g], outputs=status)
    for ctrl in (plasticity, mut_rate, mut_scale, max_g):
        ctrl.change(fn=set_evolution, inputs=[evol_on, plasticity, mut_rate, mut_scale, max_g], outputs=status)

    fps.change(fn=set_playback_fps, inputs=fps, outputs=status)

    canvas.select(fn=plant_on_click, outputs=[canvas, status])

    # Save/Load brain
    save_btn.click(fn=save_brain, outputs=save_out)
    load_in.change(fn=load_brain, inputs=load_in, outputs=status)

    # Timer
    demo.load(fn=tick, outputs=[canvas, status, brain_box])
    gr.Timer(0.03).tick(fn=tick, outputs=[canvas, status, brain_box])

if __name__ == "__main__":
    demo.launch()
