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

# Benchmark name used in [START] env= field
BENCHMARK: str = "solar-optimization-env"


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
# 3. Log helpers — EXACT format required by grader
#
# Validator expects these precise patterns (key=value, NOT JSON):
#   [START] task=<task_id> env=<benchmark> model=<model_name>
#   [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   task=<task_id> success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error) -> None:
    error_str = error if error else "null"
    done_str  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    # rewards= must be a comma-separated list of per-step values (not a sum)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    success_str = str(success).lower()
    print(
        f"[END] task={task} success={success_str} "
        f"steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


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
        f"- Sun elevation: {state.sun_elevation:.2f}deg\n"
        f"- Sun azimuth: {state.sun_azimuth:.2f}deg\n"
        f"- Panel tilt: {state.panel_tilt:.2f}deg\n"
        f"- Panel rotation: {state.panel_rotation:.2f}deg\n"
        f"- Predicted irradiance: {state.predicted_irradiance:.2f} W/m2\n\n"
        f"Respond with ONLY a JSON object like: "
        '{"tilt_change": 0.5, "rotation_change": -0.3}\n'
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
        text  = response_text.strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
            tc   = float(data.get("tilt_change", 0.0))
            rc   = float(data.get("rotation_change", 0.0))
            tc   = max(-1.0, min(1.0, tc))
            rc   = max(-1.0, min(1.0, rc))
            return SolarAction(tilt_change=tc, rotation_change=rc)
    except Exception:
        pass
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
    env: SolarEnv    = TASK_REGISTRY[task_name](seed=SEED)
    rewards:         List[float] = []
    steps_taken:     int         = 0
    # CRITICAL: initialize score to 0.001 — grader rejects exactly 0.0
    score:           float       = 0.001
    success:         bool        = False

    log_start(task=task_name, model=MODEL_NAME)

    try:
        state = env.reset()

        for step in range(1, MAX_STEPS + 1):

            # LLM call through grader proxy (required)
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
        # CRITICAL: score must be STRICTLY between 0 and 1 (exclusive)
        # Clamp to [0.001, 0.999] — grader rejects exactly 0.0 or 1.0
        score   = round(min(max(raw, 0.001), 0.999), 4)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error in {task_name}: {exc}", flush=True)
        # score stays at initialized 0.001 — never exactly 0.0

    finally:
        # Belt-and-suspenders: guarantee score is always strictly in (0, 1)
        score = round(min(max(score, 0.001), 0.999), 4)
        log_end(
            task=task_name,
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


# ---------------------------------------------------------------------------
# 7. Main — run ALL tasks in a single invocation (required by grader)
# ---------------------------------------------------------------------------

async def main() -> None:
    for task_name in ["easy", "medium", "hard"]:
        await run_task(task_name)


if __name__ == "__main__":
    asyncio.run(main())