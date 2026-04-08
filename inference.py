"""
inference.py — OpenEnv Hackathon Submission Script
Solar Panel Optimization RL Environment

Required env vars (injected by grader):
    API_BASE_URL  — LiteLLM proxy endpoint
    API_KEY       — grader-provided key (NOT HF_TOKEN)
    MODEL_NAME    — model to use for inference
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import List

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openai import OpenAI

from env.models import SolarAction, SolarState
from env.solar_env import SolarEnv
from env.tasks import TASK_REGISTRY


# ---------------------------------------------------------------------------
# 1. Environment variables — use API_KEY as required by grader
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "<your-active-endpoint>").strip()
API_KEY:      str = os.environ.get("API_KEY", "").strip()       # grader injects this
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "<your-active-model>").strip()

# Fallback: some graders use HF_TOKEN instead of API_KEY
if not API_KEY:
    API_KEY = os.environ.get("HF_TOKEN", "").strip()

if not API_KEY:
    sys.exit("ERROR: API_KEY environment variable is not set.")

LOCAL_IMAGE_NAME: str = os.environ.get("LOCAL_IMAGE_NAME", "")

# OpenAI client pointed at the grader's LiteLLM proxy
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


# ---------------------------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------------------------

SEED:                    int   = 42
MAX_STEPS:               int   = 96
SUCCESS_SCORE_THRESHOLD: float = 0.5

MAX_TOTAL_REWARD: dict[str, float] = {
    "easy":   7200.0,
    "medium": 6500.0,
    "hard":   5000.0,
}


# ---------------------------------------------------------------------------
# 3. Log helpers — exact format required by grader
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    payload = json.dumps({"task": task, "seed": SEED, "model": model})
    print(f"[START] {payload}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error) -> None:
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
    payload = json.dumps({
        "success":      success,
        "steps":        steps,
        "score":        round(score, 4),
        "total_reward": round(sum(rewards), 6),
    })
    print(f"[END] {payload}", flush=True)


# ---------------------------------------------------------------------------
# 4. LLM call — goes through the grader's proxy (required)
# ---------------------------------------------------------------------------

def get_llm_action(state: SolarState, step: int, task: str) -> str:
    """
    Ask the LLM for an action given the current solar panel state.
    This call MUST go through the grader's API_BASE_URL proxy.
    Returns a JSON string describing the action.
    """
    prompt = (
        f"You are controlling a solar panel tracker. "
        f"Current state at step {step} of task '{task}':\n"
        f"- Time of day: {state.time_of_day:.2f}h\n"
        f"- Sun elevation: {state.sun_elevation:.2f}°\n"
        f"- Sun azimuth: {state.sun_azimuth:.2f}°\n"
        f"- Panel tilt: {state.panel_tilt:.2f}°\n"
        f"- Panel rotation: {state.panel_rotation:.2f}°\n"
        f"- Predicted irradiance: {state.predicted_irradiance:.2f} W/m²\n\n"
        f"Respond with ONLY a JSON object like: "
        f'{{\"tilt_change\": 0.5, \"rotation_change\": -0.3}}\n'
        f"Values must be between -1.0 and 1.0."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return '{"tilt_change": 0.0, "rotation_change": 0.0}'


def parse_llm_action(response_text: str, state: SolarState, config) -> SolarAction:
    """
    Parse LLM response into SolarAction.
    Falls back to greedy policy if parsing fails.
    """
    try:
        # Extract JSON from response
        text = response_text.strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
            tc = float(data.get("tilt_change", 0.0))
            rc = float(data.get("rotation_change", 0.0))
            tc = max(-1.0, min(1.0, tc))
            rc = max(-1.0, min(1.0, rc))
            return SolarAction(tilt_change=tc, rotation_change=rc)
    except Exception:
        pass
    # Fallback to greedy
    return greedy_action(state, config)


# ---------------------------------------------------------------------------
# 5. Greedy fallback policy
# ---------------------------------------------------------------------------

def greedy_action(state: SolarState, config) -> SolarAction:
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
# 6. Episode runner
# ---------------------------------------------------------------------------

async def run_task(task_name: str) -> None:
    env: SolarEnv = TASK_REGISTRY[task_name](seed=SEED)
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.001  # never exactly 0.0 — grader requires strictly > 0
    success:     bool        = False

    log_start(task=task_name, model=MODEL_NAME)

    try:
        state = env.reset()

        for step in range(1, MAX_STEPS + 1):

            # --- LLM call through grader proxy (required) ---
            llm_response = get_llm_action(state, step, task_name)
            action       = parse_llm_action(llm_response, state, env.config)
            action_str   = json.dumps({
                "tilt_change":     round(action.tilt_change, 4),
                "rotation_change": round(action.rotation_change, 4),
            })

            result      = env.step(action)
            reward      = result.reward
            done        = result.done

            rewards.append(reward)
            steps_taken = step
            state       = result.state

            log_step(step=step, action=action_str,
                     reward=reward, done=done, error=None)

            if done:
                break

        max_reward = MAX_TOTAL_REWARD.get(task_name, 7200.0)
        raw        = sum(rewards) / max_reward if max_reward > 0 else 0.0
        # Must be STRICTLY between 0 and 1 (exclusive) — grader rejects 0.0 and 1.0
        score      = round(min(max(raw, 0.001), 0.999), 4)
        success    = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        # Final safety clamp — ensures score is never exactly 0.0 or 1.0
        score = round(min(max(score, 0.001), 0.999), 4)
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

async def main() -> None:
    for task_name in ["easy", "medium", "hard"]:
        await run_task(task_name)


if __name__ == "__main__":
    asyncio.run(main())