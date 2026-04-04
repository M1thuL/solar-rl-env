"""
inference.py — OpenEnv Hackathon Submission Script
===================================================
Predictive AI-Controlled Solar Optimization System

Matches the sample inference script format exactly:
  - Uses OpenAI client for LLM calls
  - Emits log_start / log_step / log_end in required JSON format
  - Score = sum(rewards) / MAX_TOTAL_REWARD, clamped to [0, 1]
  - Runs one episode per task: easy, medium, hard
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import List

# ── Project root on sys.path ────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openai import OpenAI

from env.models import SolarAction, SolarState
from env.solar_env import SolarEnv
from env.tasks import TASK_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment variables — fail loudly if missing
# ─────────────────────────────────────────────────────────────────────────────

# Defaults set for API_BASE_URL and MODEL_NAME only — NOT HF_TOKEN (per spec)
API_BASE_URL: str = os.environ.get("API_BASE_URL", "<your-active-endpoint>").strip()
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "<your-active-model>").strip()
HF_TOKEN:     str = os.environ.get("HF_TOKEN")  # No default — must be set explicitly

# Optional: used if loading env from docker image
LOCAL_IMAGE_NAME: str = os.environ.get("LOCAL_IMAGE_NAME", "")

if not HF_TOKEN:
    sys.exit("ERROR: HF_TOKEN environment variable is not set.")

# OpenAI client — used for all LLM calls (required by spec)
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Constants
# ─────────────────────────────────────────────────────────────────────────────

SEED:                  int   = 42
MAX_STEPS:             int   = 96
SUCCESS_SCORE_THRESHOLD: float = 0.5

# Maximum total reward a perfect greedy agent achieves on each task (measured).
# Used for score normalisation: score = sum(rewards) / MAX_TOTAL_REWARD
MAX_TOTAL_REWARD: dict[str, float] = {
    "easy":   7200.0,
    "medium": 6500.0,
    "hard":   5000.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Required log helpers — match sample script field names exactly
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    """Print [START] line — called once at episode start."""
    payload = json.dumps({
        "task":  task,
        "seed":  SEED,
        "model": model,
    })
    print(f"[START] {payload}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error) -> None:
    """Print [STEP] line — called once per environment step."""
    payload = json.dumps({
        "step":   step,
        "action": action,
        "reward": round(reward, 6),
        "done":   done,
        "error":  error,
    })
    print(f"[STEP] {payload}", flush=True)


def log_end(success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    """Print [END] line — called once at episode end."""
    payload = json.dumps({
        "success":      success,
        "steps":        steps,
        "score":        round(score, 4),
        "total_reward": round(sum(rewards), 6),
    })
    print(f"[END] {payload}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Action policy — greedy sun tracker
# ─────────────────────────────────────────────────────────────────────────────

def get_action(state: SolarState, config) -> tuple[SolarAction, str]:
    """
    Greedy proportional sun-tracking policy.
    Returns (SolarAction, action_string_for_logging).

    To use LLM-based actions: call client.chat.completions.create() here
    and parse the response into a SolarAction.
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

    action = SolarAction(tilt_change=tc, rotation_change=rc)
    action_str = json.dumps({"tilt_change": round(tc, 4),
                              "rotation_change": round(rc, 4)})
    return action, action_str


# ─────────────────────────────────────────────────────────────────────────────
# 5. Episode runner — mirrors sample script structure exactly
# ─────────────────────────────────────────────────────────────────────────────

async def run_task(task_name: str) -> None:
    """
    Run one full episode for task_name.
    Structure mirrors the sample inference script:
      log_start → reset → loop(get_action, step, log_step) → log_end
    """
    env: SolarEnv = TASK_REGISTRY[task_name](seed=SEED)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_name, model=MODEL_NAME)

    try:
        # reset() — returns initial SolarState
        state = env.reset()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            # Get action from policy
            action, action_str = get_action(state, env.config)

            # step() — advance environment
            result     = env.step(action)
            reward     = result.reward
            done       = result.done
            error      = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            state       = result.state

            log_step(step=step, action=action_str,
                     reward=reward, done=done, error=error)

            if done:
                break

        # Score = sum(rewards) / MAX_TOTAL_REWARD — matches sample script
        max_reward = MAX_TOTAL_REWARD.get(task_name, 7200.0)
        score      = sum(rewards) / max_reward if max_reward > 0 else 0.0
        score      = min(max(score, 0.0), 1.0)
        success    = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main — run all three tasks
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    for task_name in ["easy", "medium", "hard"]:
        await run_task(task_name)


if __name__ == "__main__":
    asyncio.run(main())