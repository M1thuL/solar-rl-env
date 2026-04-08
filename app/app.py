"""
app.py — Gradio demo + FastAPI OpenEnv HTTP endpoints.

The grader calls:
  POST /reset        → returns SolarState JSON
  POST /step         → accepts SolarAction JSON, returns StepResult JSON
  GET  /state        → returns current SolarState JSON

Gradio UI runs on the same port via mounted ASGI app.
"""

from __future__ import annotations

import sys
import os

_here         = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_here)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.models import SolarAction, SolarState
from env.solar_env import SolarEnv
from env.tasks import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Global env instance — shared between API calls
# ---------------------------------------------------------------------------

_env: SolarEnv = TASK_REGISTRY["easy"](seed=42)
_current_state: SolarState = _env.reset()


# ---------------------------------------------------------------------------
# FastAPI app with OpenEnv endpoints
# ---------------------------------------------------------------------------

api = FastAPI(title="Solar RL Environment API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    tilt_change: float = 0.0
    rotation_change: float = 0.0


MAX_REWARD_PER_TASK = {"easy": 7200.0, "medium": 6500.0, "hard": 5000.0}

@api.post("/reset")
def api_reset(body: ResetRequest = None):
    global _env, _current_state
    task = body.task if body else "easy"
    seed = body.seed if body else 42
    task = task if task in TASK_REGISTRY else "easy"
    _env = TASK_REGISTRY[task](seed=seed)
    _current_state = _env.reset()
    result = _current_state.model_dump()
    result["task"] = task
    result["seed"] = seed
    result["max_reward"] = MAX_REWARD_PER_TASK.get(task, 7200.0)
    return result


@api.post("/step")
def api_step(body: StepRequest):
    global _current_state
    action = SolarAction(
        tilt_change=body.tilt_change,
        rotation_change=body.rotation_change,
    )
    result = _env.step(action)
    _current_state = result.state

    # Compute normalised score strictly in (0, 1) — never 0.0 or 1.0
    task    = _env.task.value
    ceiling = MAX_REWARD_PER_TASK.get(task, 7200.0)
    raw     = result.info.episode_cumulative_energy / ceiling
    score   = round(min(max(raw, 0.001), 0.999), 4)

    return {
        "state":      result.state.model_dump(),
        "reward":     result.reward,
        "terminated": result.terminated,
        "truncated":  result.truncated,
        "done":       result.done,
        "score":      score,
        "info":       result.info.model_dump(),
    }


@api.get("/state")
def api_state():
    return _current_state.model_dump()


@api.get("/health")
def health():
    return {"status": "ok"}


@api.get("/score")
def api_score():
    """Return current episode score strictly in (0, 1)."""
    max_reward = {"easy": 7200.0, "medium": 6500.0, "hard": 5000.0}
    task = _env.task.value if hasattr(_env, "task") else "easy"
    ceiling = max_reward.get(task, 7200.0)
    raw = _env._cumulative_energy / ceiling if ceiling > 0 else 0.0
    score = round(min(max(raw, 0.001), 0.999), 4)
    return {"score": score, "task": task}


# ---------------------------------------------------------------------------
# Greedy policy + simulation (for Gradio UI)
# ---------------------------------------------------------------------------

def _greedy_action(state: SolarState, config) -> SolarAction:
    if state.sun_elevation <= 0.0:
        err_tilt = 0.0   - state.panel_tilt
        err_rot  = 180.0 - state.panel_rotation
    else:
        err_tilt = state.sun_elevation - state.panel_tilt
        err_rot  = state.sun_azimuth   - state.panel_rotation
    err_rot = (err_rot + 180.0) % 360.0 - 180.0
    tc = max(-1.0, min(1.0, err_tilt / config.max_tilt_step))
    rc = max(-1.0, min(1.0, err_rot  / config.max_rotation_step))
    return SolarAction(tilt_change=tc, rotation_change=rc)


def _static_action(state: SolarState, config) -> SolarAction:
    return SolarAction(tilt_change=0.0, rotation_change=0.0)


def _run_episode(task_name, seed, policy_fn):
    env = TASK_REGISTRY[task_name](seed=int(seed))
    state = env.reset()
    times, energies, tilts, sun_elevations, irradiances = [], [], [], [], []
    total_reward = 0.0
    while True:
        action = policy_fn(state, env.config)
        result = env.step(action)
        s = result.state
        times.append(s.time_of_day)
        energies.append(s.energy_this_step)
        tilts.append(s.panel_tilt)
        sun_elevations.append(s.sun_elevation)
        irradiances.append(s.true_irradiance)
        total_reward += result.reward
        state = s
        if result.done:
            break
    return {
        "times": times, "energies": energies, "tilts": tilts,
        "sun_elevations": sun_elevations, "irradiances": irradiances,
        "total_energy": result.info.episode_cumulative_energy,
        "total_reward": total_reward, "steps": len(times),
    }


def run_simulation(task_name, seed):
    static = _run_episode(task_name, seed, _static_action)
    greedy = _run_episode(task_name, seed, _greedy_action)
    return static, greedy


TASK_COLORS = {"easy": "#f59e0b", "medium": "#3b82f6", "hard": "#ef4444"}
STATIC_COLOR = "#6b7280"
GREEDY_COLOR = "#10b981"


def build_figure(static, greedy, task):
    bg, panel, muted, grid_c, sun_c = "#111827", "#1f2937", "#9ca3af", "#374151", "#f59e0b"

    fig = plt.figure(figsize=(12, 10), facecolor=bg)
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    def night(ax):
        ax.axvspan(0, 6, alpha=0.12, color="#1e40af")
        ax.axvspan(18, 24, alpha=0.12, color="#1e40af")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.tick_params(colors=muted)
        for sp in ax.spines.values():
            sp.set_edgecolor(grid_c)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(panel)
    ax1.fill_between(greedy["times"], greedy["energies"], alpha=0.15, color=GREEDY_COLOR)
    ax1.fill_between(static["times"], static["energies"], alpha=0.15, color=STATIC_COLOR)
    ax1.plot(greedy["times"], greedy["energies"], color=GREEDY_COLOR, linewidth=2,
             label=f"Greedy  ({greedy['total_energy']:.2f} Wh)")
    ax1.plot(static["times"], static["energies"], color=STATIC_COLOR, linewidth=2,
             linestyle="--", label=f"Static  ({static['total_energy']:.2f} Wh)")
    night(ax1)
    ax1.set_title("Energy harvested — Static vs Greedy", color="white", fontsize=11, pad=8)
    ax1.set_xlabel("Hour of day", color=muted, fontsize=9)
    ax1.set_ylabel("Energy (Wh)", color=muted, fontsize=9)
    ax1.legend(loc="upper right", fontsize=8, facecolor=grid_c, labelcolor="white", framealpha=0.9)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(panel)
    ax2.plot(greedy["times"], greedy["sun_elevations"], color=sun_c, linewidth=2,
             linestyle=":", label="Sun elevation (°)", alpha=0.8)
    ax2.plot(greedy["times"], greedy["tilts"], color=GREEDY_COLOR, linewidth=2, label="Greedy tilt (°)")
    ax2.plot(static["times"], static["tilts"], color=STATIC_COLOR, linewidth=2,
             linestyle="--", label="Static tilt (°)")
    ax2.axhline(0, color=grid_c, linewidth=0.8, linestyle=":")
    night(ax2)
    ax2.set_title("Panel tilt vs sun elevation", color="white", fontsize=11, pad=8)
    ax2.set_xlabel("Hour of day", color=muted, fontsize=9)
    ax2.set_ylabel("Degrees (°)", color=muted, fontsize=9)
    ax2.legend(loc="upper right", fontsize=8, facecolor=grid_c, labelcolor="white", framealpha=0.9)

    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(panel)
    cum_g, cum_s, rg, rs = [], [], 0.0, 0.0
    for g, s in zip(greedy["energies"], static["energies"]):
        rg += g; rs += s
        cum_g.append(rg); cum_s.append(rs)
    gap = [g - s for g, s in zip(cum_g, cum_s)]
    ax3.fill_between(greedy["times"], gap, alpha=0.2, color=GREEDY_COLOR)
    ax3.plot(greedy["times"], gap, color=GREEDY_COLOR, linewidth=2, label="Greedy advantage (Wh)")
    ax3.axhline(0, color=grid_c, linewidth=0.8, linestyle=":")
    night(ax3)
    ax3.set_title("Cumulative energy advantage — Greedy over Static", color="white", fontsize=11, pad=8)
    ax3.set_xlabel("Hour of day", color=muted, fontsize=9)
    ax3.set_ylabel("Cumulative Δ energy (Wh)", color=GREEDY_COLOR, fontsize=9)
    ax3.legend(loc="upper left", fontsize=8, facecolor=grid_c, labelcolor="white", framealpha=0.9)

    fig.tight_layout(pad=2.0)
    return fig


def on_run(task, seed):
    static, greedy = run_simulation(task, seed)
    emoji = {"easy": "☀️", "medium": "⛅", "hard": "🌩️"}.get(task, "")
    s_energy = static["total_energy"]
    g_energy = greedy["total_energy"]
    THRESHOLD = 0.001
    if s_energy < THRESHOLD:
        improvement_str = "N/A (static baseline produces zero energy)"
    else:
        pct  = (g_energy - s_energy) / s_energy * 100
        sign = "+" if pct >= 0 else ""
        improvement_str = f"**{sign}{pct:.2f}%**"
    reward_delta = greedy["total_reward"] - static["total_reward"]
    reward_sign  = "+" if reward_delta >= 0 else ""
    metrics_md = f"""
### {emoji} Comparison — **{task.upper()}** task  ·  seed `{int(seed)}`

|  | Static panel | Greedy tracker | Improvement |
|--|:---:|:---:|:---:|
| **Total energy (Wh)** | {s_energy:.3f} | {g_energy:.3f} | {improvement_str} |
| **Total reward** | {static['total_reward']:.3f} | {greedy['total_reward']:.3f} | {reward_sign}{reward_delta:.3f} |
| **Steps** | {static['steps']} | {greedy['steps']} | — |

> Static panel = tilt 0°, rotation 0°, never moves.
> Greedy tracker = always points directly at the sun.
    """.strip()
    fig = build_figure(static, greedy, task)
    plt.close("all")
    return metrics_md, fig


def build_ui():
    with gr.Blocks(title="Solar Panel Optimization — RL Demo") as demo:
        gr.Markdown("""
# Solar Panel Optimization — RL Environment Demo
Comparing **Static panel** vs **Greedy tracker**. A trained RL agent should outperform both.
        """)
        with gr.Row():
            task_dropdown = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy",
                                        label="Task difficulty")
            seed_slider = gr.Slider(minimum=1, maximum=100, value=42, step=1, label="Random seed")
            run_button = gr.Button("▶  Run comparison", variant="primary", scale=1)
        metrics_box = gr.Markdown("*Select a task and click **▶ Run comparison**.*")
        chart_out = gr.Plot(label="Static vs Greedy comparison")
        with gr.Accordion("How this environment works", open=False):
            gr.Markdown("""
**State:** `time_of_day` · `sun_azimuth` · `sun_elevation` · `predicted_irradiance` · `panel_tilt` · `panel_rotation`

**Actions:** `tilt_change ∈ [-1,1]` → ±5°/step · `rotation_change ∈ [-1,1]` → ±10°/step

**Reward:** `energy_harvested − movement_cost − misalignment_penalty`

**API endpoints:** `POST /reset` · `POST /step` · `GET /state`
            """)
        run_button.click(fn=on_run, inputs=[task_dropdown, seed_slider],
                         outputs=[metrics_box, chart_out])
    return demo


# ---------------------------------------------------------------------------
# Entry point — mount Gradio on FastAPI
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    gradio_app = build_ui()
    app = gr.mount_gradio_app(api, gradio_app, path="/")
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    uvicorn.run(app, host=server_name, port=7860)


if __name__ == "__main__":
    main()