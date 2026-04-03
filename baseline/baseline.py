"""
baseline.py — Greedy sun-tracking baseline agent.

Strategy
--------
At every step, read the current sun position from the state and compute
the angular error between the panel and the sun.  Convert that error into
a normalised action in [-1, 1] and submit it.

This is the simplest possible policy: no learning, no memory, no model.
It serves two purposes:
  1. Sanity check  — if the env is working correctly, this agent should
                     harvest significantly more energy than random actions.
  2. Performance floor — any trained RL agent should beat this score.

Usage
-----
    # Run on all three tasks (default):
    python baseline/baseline.py

    # Run on a single task:
    python baseline/baseline.py --task easy
    python baseline/baseline.py --task medium
    python baseline/baseline.py --task hard

    # Change the RNG seed:
    python baseline/baseline.py --seed 7
"""

from __future__ import annotations

import argparse
import sys
import os

# Make sure the project root is on sys.path so `env` package resolves
# whether baseline.py is run from the project root or the baseline/ dir.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.models import SolarAction, SolarState
from env.solar_env import SolarEnv
from env.tasks import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Greedy policy
# ---------------------------------------------------------------------------

def greedy_action(state: SolarState, config) -> SolarAction:
    """
    Compute the greedy action that moves the panel toward the sun.

    The policy is intentionally simple:

        error_tilt     = sun_elevation - panel_tilt
        error_rotation = shortest_arc(sun_azimuth - panel_rotation)

    We divide each error by the maximum step size to get a value in [-1, 1].
    This means:
      - If the error is larger than one max-step, the agent moves at full speed.
      - If the error is smaller (panel nearly aligned), the agent moves gently.

    This gives naturally smooth, proportional movement — no jerky snapping.

    Parameters
    ----------
    state  : SolarState   — current observation
    config : EpisodeConfig — needed for max_tilt_step and max_rotation_step

    Returns
    -------
    SolarAction with tilt_change and rotation_change in [-1, 1]
    """

    # ── Tilt error ───────────────────────────────────────────────────────
    # How far is the panel from facing the sun vertically?
    # Positive → panel needs to tilt up.  Negative → tilt down.
    error_tilt = state.sun_elevation - state.panel_tilt

    # ── Rotation error (shortest arc) ───────────────────────────────────
    # Raw difference can be up to 359° the "long way round".
    # The ((x + 180) % 360 - 180) trick maps it to [-180, +180],
    # always choosing the shorter path around the compass.
    raw_rotation_error = state.sun_azimuth - state.panel_rotation
    error_rotation = (raw_rotation_error + 180.0) % 360.0 - 180.0

    # ── Normalise to [-1, 1] ─────────────────────────────────────────────
    # Dividing by max_step converts degrees → fraction of max movement.
    # clip() ensures we never exceed [-1, 1] even for large errors.
    tilt_change = error_tilt / config.max_tilt_step
    tilt_change = max(-1.0, min(1.0, tilt_change))

    rotation_change = error_rotation / config.max_rotation_step
    rotation_change = max(-1.0, min(1.0, rotation_change))

    # ── Night behaviour ──────────────────────────────────────────────────
    # When the sun is below the horizon, there is nothing to track.
    # We park the panel flat and facing South (180°) — a safe resting
    # position that pre-orients for the morning sun.
    if state.sun_elevation <= 0.0:
        # Target: tilt=0 (flat), rotation=180 (South)
        park_tilt_error     = 0.0   - state.panel_tilt
        park_rotation_error = 180.0 - state.panel_rotation
        park_rotation_error = (park_rotation_error + 180.0) % 360.0 - 180.0

        tilt_change     = max(-1.0, min(1.0, park_tilt_error     / config.max_tilt_step))
        rotation_change = max(-1.0, min(1.0, park_rotation_error / config.max_rotation_step))

    return SolarAction(tilt_change=tilt_change, rotation_change=rotation_change)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env: SolarEnv, verbose: bool = False) -> dict:
    """
    Run one complete episode with the greedy policy.

    Parameters
    ----------
    env     : SolarEnv  — already constructed, not yet reset
    verbose : bool      — if True, print a row per step

    Returns
    -------
    dict with keys:
        total_reward  (float) — sum of all step rewards
        total_energy  (float) — cumulative energy in Wh
        steps         (int)   — number of steps taken
        task          (str)   — task difficulty name
    """

    # ── Initialise ───────────────────────────────────────────────────────
    state = env.reset()
    total_reward = 0.0
    steps = 0

    if verbose:
        print(f"\n{'step':>5}  {'time':>5}  {'sun_el':>7}  "
              f"{'tilt':>6}  {'rot':>6}  {'energy':>8}  {'reward':>8}")
        print("-" * 62)

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        # Compute greedy action from current state
        action = greedy_action(state, env.config)

        # Step the environment
        result = env.step(action)

        # Accumulate reward
        total_reward += result.reward
        steps += 1

        if verbose:
            print(
                f"{steps:>5}  "
                f"{state.time_of_day:>5.2f}  "
                f"{state.sun_elevation:>7.2f}  "
                f"{result.state.panel_tilt:>6.2f}  "
                f"{result.state.panel_rotation:>6.2f}  "
                f"{result.state.energy_this_step:>8.4f}  "
                f"{result.reward:>8.4f}"
            )

        # Advance state
        state = result.state

        # Episode over?
        if result.done:
            break

    return {
        "total_reward": total_reward,
        "total_energy": result.info.episode_cumulative_energy,
        "steps":        steps,
        "task":         env.task.value,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results(results: dict) -> None:
    """Print a clean summary of one episode's results."""
    task  = results["task"].upper()
    width = 38

    print(f"\n  {'─' * width}")
    print(f"  {'Baseline Results':^{width}}")
    print(f"  {'Task: ' + task:^{width}}")
    print(f"  {'─' * width}")
    print(f"  {'Total reward':<22} {results['total_reward']:>10.4f}")
    print(f"  {'Total energy (Wh)':<22} {results['total_energy']:>10.4f}")
    print(f"  {'Steps taken':<22} {results['steps']:>10d}")
    print(f"  {'─' * width}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the greedy baseline agent on the Solar RL Environment."
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task tier to run.  Default: all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility.  Default: 42",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a row per step showing sun position and energy.",
    )
    args = parser.parse_args()

    # Determine which tasks to run
    tasks_to_run = (
        ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    )

    print(f"\n  Greedy baseline  |  seed={args.seed}")

    for task_name in tasks_to_run:
        # Build env via registry (no need to import each factory explicitly)
        env = TASK_REGISTRY[task_name](seed=args.seed)

        # Run one full episode
        results = run_episode(env, verbose=args.verbose)

        # Show summary
        print_results(results)


if __name__ == "__main__":
    main()