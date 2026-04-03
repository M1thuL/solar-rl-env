"""
app.py — Gradio demo with Static vs Greedy baseline comparison.

Changes from previous version:
  1. Added _static_action()  — panel never moves (zero action every step)
  2. run_simulation() now runs BOTH agents on the SAME environment seed
     and returns data for both side-by-side
  3. build_figure() updated to overlay both agents on every chart
  4. on_run() updated metrics table to compare Static vs Greedy
  5. UI unchanged — same controls, one button, one chart output
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

from env.models import SolarAction, SolarState
from env.solar_env import SolarEnv
from env.tasks import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Two policies
# ---------------------------------------------------------------------------

def _static_action(state: SolarState, config) -> SolarAction:
    """
    Static baseline: never move the panel.
    Panel stays at its initial pose (tilt=0, rotation=0) for the entire day.
    This represents a fixed-mount panel — the simplest possible installation.
    """
    return SolarAction(tilt_change=0.0, rotation_change=0.0)


def _greedy_action(state: SolarState, config) -> SolarAction:
    """
    Greedy baseline: always point directly at the sun.
    At night, park flat facing South to pre-orient for sunrise.
    """
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


# ---------------------------------------------------------------------------
# Episode runner — runs ONE policy on a fresh env with given seed
# ---------------------------------------------------------------------------

def _run_episode(task_name: str, seed: int, policy_fn) -> dict:
    """
    Run a single episode with the given policy function.
    Both agents use the SAME seed so they face identical cloud conditions.
    """
    env: SolarEnv = TASK_REGISTRY[task_name](seed=int(seed))
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
        "times":          times,
        "energies":       energies,
        "tilts":          tilts,
        "sun_elevations": sun_elevations,
        "irradiances":    irradiances,
        "total_energy":   result.info.episode_cumulative_energy,
        "total_reward":   total_reward,
        "steps":          len(times),
    }


def run_simulation(task_name: str, seed: int) -> tuple[dict, dict]:
    """
    Run BOTH policies and return (static_data, greedy_data).
    Same seed = same random cloud draws = fair comparison.
    """
    static = _run_episode(task_name, seed, _static_action)
    greedy = _run_episode(task_name, seed, _greedy_action)
    return static, greedy


# ---------------------------------------------------------------------------
# Chart builder — overlays both agents on each subplot
# ---------------------------------------------------------------------------

TASK_COLORS = {"easy": "#f59e0b", "medium": "#3b82f6", "hard": "#ef4444"}

# Fixed colors for each agent — consistent regardless of task
STATIC_COLOR = "#6b7280"   # grey  — static panel
GREEDY_COLOR = "#10b981"   # green — greedy tracker

def build_figure(static: dict, greedy: dict, task: str) -> plt.Figure:
    bg     = "#111827"
    panel  = "#1f2937"
    muted  = "#9ca3af"
    grid_c = "#374151"
    sun_c  = "#f59e0b"

    fig = plt.figure(figsize=(12, 10), facecolor=bg)
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    # ── Shared night shading helper ──────────────────────────────────────
    def night(ax):
        ax.axvspan(0,  6,  alpha=0.12, color="#1e40af")
        ax.axvspan(18, 24, alpha=0.12, color="#1e40af")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.tick_params(colors=muted)
        for sp in ax.spines.values():
            sp.set_edgecolor(grid_c)

    # ── Chart 1: Energy comparison ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(panel)

    ax1.fill_between(greedy["times"], greedy["energies"],
                     alpha=0.15, color=GREEDY_COLOR)
    ax1.fill_between(static["times"], static["energies"],
                     alpha=0.15, color=STATIC_COLOR)

    ax1.plot(greedy["times"], greedy["energies"],
             color=GREEDY_COLOR, linewidth=2,
             label=f"Greedy  ({greedy['total_energy']:.2f} Wh)")
    ax1.plot(static["times"], static["energies"],
             color=STATIC_COLOR, linewidth=2, linestyle="--",
             label=f"Static  ({static['total_energy']:.2f} Wh)")

    night(ax1)
    ax1.set_title("Energy harvested per step — Static vs Greedy",
                  color="white", fontsize=11, pad=8)
    ax1.set_xlabel("Hour of day", color=muted, fontsize=9)
    ax1.set_ylabel("Energy (Wh)", color=muted, fontsize=9)
    ax1.legend(loc="upper right", fontsize=8,
               facecolor=grid_c, labelcolor="white", framealpha=0.9)

    # ── Chart 2: Panel tilt comparison ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(panel)

    ax2.plot(greedy["times"], greedy["sun_elevations"],
             color=sun_c, linewidth=2, linestyle=":",
             label="Sun elevation (°)", alpha=0.8)
    ax2.plot(greedy["times"], greedy["tilts"],
             color=GREEDY_COLOR, linewidth=2, label="Greedy tilt (°)")
    ax2.plot(static["times"], static["tilts"],
             color=STATIC_COLOR, linewidth=2, linestyle="--",
             label="Static tilt (°)")

    ax2.axhline(0, color=grid_c, linewidth=0.8, linestyle=":")
    night(ax2)
    ax2.set_title("Panel tilt vs sun elevation",
                  color="white", fontsize=11, pad=8)
    ax2.set_xlabel("Hour of day", color=muted, fontsize=9)
    ax2.set_ylabel("Degrees (°)", color=muted, fontsize=9)
    ax2.legend(loc="upper right", fontsize=8,
               facecolor=grid_c, labelcolor="white", framealpha=0.9)

    # ── Chart 3: Cumulative energy gap ───────────────────────────────────
    # Shows how the greedy advantage builds up over the day.
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(panel)

    cum_greedy = []
    cum_static = []
    run_g = run_s = 0.0
    for g, s in zip(greedy["energies"], static["energies"]):
        run_g += g
        run_s += s
        cum_greedy.append(run_g)
        cum_static.append(run_s)

    gap = [g - s for g, s in zip(cum_greedy, cum_static)]

    ax3.fill_between(greedy["times"], gap, alpha=0.2, color=GREEDY_COLOR)
    ax3.plot(greedy["times"], gap,
             color=GREEDY_COLOR, linewidth=2,
             label="Greedy advantage (Wh)")
    ax3.axhline(0, color=grid_c, linewidth=0.8, linestyle=":")

    night(ax3)
    ax3.set_title("Cumulative energy advantage — Greedy over Static",
                  color="white", fontsize=11, pad=8)
    ax3.set_xlabel("Hour of day", color=muted, fontsize=9)
    ax3.set_ylabel("Cumulative Δ energy (Wh)", color=GREEDY_COLOR, fontsize=9)
    ax3.legend(loc="upper left", fontsize=8,
               facecolor=grid_c, labelcolor="white", framealpha=0.9)

    fig.tight_layout(pad=2.0)
    return fig


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def on_run(task: str, seed: int):
    static, greedy = run_simulation(task, seed)
    emoji = {"easy": "☀️", "medium": "⛅", "hard": "🌩️"}.get(task, "")

    # Energy improvement percentage
    # Static panel at tilt=0 can harvest zero energy (sun never hits the face).
    # Never divide by zero — show a clear message instead of inf/nan/9999%.
    s_energy = static["total_energy"]
    g_energy = greedy["total_energy"]

    # Use a small threshold (0.001 Wh) rather than exact zero.
    # A flat panel may harvest a tiny near-zero amount at dawn/dusk due to
    # grazing irradiance — this looks like 0.000 when rounded to 3dp but
    # is still a valid (microscopic) denominator that produces absurd %.
    THRESHOLD = 0.001  # Wh — anything below this is "effectively zero"

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
> A trained RL agent should outperform both.
    """.strip()

    fig = build_figure(static, greedy, task)
    plt.close("all")
    return metrics_md, fig


# ---------------------------------------------------------------------------
# UI — minimal changes: updated header and description only
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Solar Panel Optimization — RL Demo") as demo:

        gr.Markdown("""
# Solar Panel Optimization — RL Environment Demo
Comparing **Static panel** (fixed mount, never moves) vs **Greedy tracker** (always faces the sun).
A trained RL agent should outperform both by handling clouds and movement costs efficiently.
        """)

        with gr.Row():
            task_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Task difficulty",
                info="Easy = clear sky · Medium = mild clouds · Hard = heavy clouds + noisy forecast",
            )
            seed_slider = gr.Slider(
                minimum=1, maximum=100, value=42, step=1,
                label="Random seed",
                info="Both agents use the same seed — identical weather conditions.",
            )
            run_button = gr.Button("▶  Run comparison", variant="primary", scale=1)

        metrics_box = gr.Markdown(
            "*Select a task and click **▶ Run comparison**.*"
        )

        chart_out = gr.Plot(label="Static vs Greedy comparison")

        with gr.Accordion("How this environment works", open=False):
            gr.Markdown("""
**State** observed each step:
`time_of_day` · `sun_azimuth` · `sun_elevation` · `predicted_irradiance` · `panel_tilt` · `panel_rotation`

**Actions:**
`tilt_change ∈ [-1,1]` → ±5°/step · `rotation_change ∈ [-1,1]` → ±10°/step

**Reward:**
```
reward = energy_harvested − movement_cost − misalignment_penalty
```
**Energy formula:**
```
energy = irradiance × cos(angle_diff) × step_duration_hours
```
**Static agent:** submits `(0, 0)` every step — panel never moves from its initial flat pose.

**Greedy agent:** computes the angular error to the sun and submits a proportional action.
            """)

        with gr.Accordion("Task parameter reference", open=False):
            gr.Markdown("""
| Parameter | Easy | Medium | Hard |
|-----------|:----:|:------:|:----:|
| Cloud noise σ | 0.0 | 0.2 | 0.5 |
| Movement penalty | 0.0 | 0.1 | 0.3 |
| Forecast noise σ | 0.0 | 0.1 | 0.3 |
| Steps / episode | 96 | 96 | 96 |
            """)

        run_button.click(
            fn=on_run,
            inputs=[task_dropdown, seed_slider],
            outputs=[metrics_box, chart_out],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    app = build_ui()
    app.launch(
        server_name=server_name,
        server_port=7860,
        share=False,
    )