"""
inference.py — OpenEnv Hackathon Submission
Solar Panel Optimization RL Environment
"""

from __future__ import annotations

import os
import sys
import requests
from openai import OpenAI

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Environment variables — exactly as injected by validator
# ---------------------------------------------------------------------------

API_KEY      = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL      = os.environ.get("ENV_URL", "https://m1thul-solar-rl-env.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Tasks — must match task ids in openenv.yaml exactly
# ---------------------------------------------------------------------------

TASKS = [
    ("easy",   "easy"),
    ("medium", "medium"),
    ("hard",   "hard"),
]

MAX_REWARD = {"easy": 7200.0, "medium": 6500.0, "hard": 5000.0}
MAX_STEPS  = 96
BENCHMARK  = "solar-rl-env"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for task_id, difficulty in TASKS:
        rewards  = []
        steps    = 0
        score    = 0.001
        success  = False

        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        try:
            # Reset environment via HTTP
            reset_resp = requests.post(
                f"{ENV_URL}/reset",
                json={"task": task_id, "seed": 42},
                timeout=30,
            ).json()

            obs  = reset_resp
            done = False

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                # LLM call through validator proxy
                prompt = (
                    f"Control a solar panel tracker. Step {step}, task '{task_id}'.\n"
                    f"Sun elevation: {obs.get('sun_elevation', 0):.1f}°, "
                    f"Sun azimuth: {obs.get('sun_azimuth', 180):.1f}°, "
                    f"Panel tilt: {obs.get('panel_tilt', 0):.1f}°, "
                    f"Panel rotation: {obs.get('panel_rotation', 0):.1f}°.\n"
                    f"Reply with ONLY: tilt_change=<-1 to 1> rotation_change=<-1 to 1>"
                )

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=30,
                        temperature=0.0,
                    )
                    llm_out = completion.choices[0].message.content.strip()
                except Exception as e:
                    print(f"[DEBUG] LLM error step {step}: {e}", flush=True)
                    llm_out = "tilt_change=0.5 rotation_change=0.3"

                # Parse LLM output
                tc, rc = 0.5, 0.3
                try:
                    for part in llm_out.replace(",", " ").split():
                        if "tilt_change=" in part:
                            tc = float(part.split("=")[1])
                        elif "rotation_change=" in part:
                            rc = float(part.split("=")[1])
                    tc = max(-1.0, min(1.0, tc))
                    rc = max(-1.0, min(1.0, rc))
                except Exception:
                    pass

                # Step environment via HTTP
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json={"tilt_change": tc, "rotation_change": rc},
                    timeout=30,
                ).json()

                reward = float(step_resp.get("reward", 0.0) or 0.0)
                done   = bool(step_resp.get("done", False))
                obs    = step_resp.get("state", step_resp)
                error  = step_resp.get("error", None)

                rewards.append(reward)
                steps = step

                error_str = str(error) if error else "null"
                action_str = f"tilt={tc:.2f},rot={rc:.2f}"
                print(
                    f"[STEP] step={step} action={action_str} "
                    f"reward={reward:.4f} done={str(done).lower()} error={error_str}",
                    flush=True,
                )

                if done:
                    break

            # Compute score strictly in (0, 1)
            ceiling = MAX_REWARD.get(task_id, 7200.0)
            raw     = sum(rewards) / ceiling if ceiling > 0 else 0.0
            score   = max(0.001, min(0.999, raw))
            # Extra safety: ensure not exactly boundary after float ops
            if score <= 0.0:
                score = 0.001
            if score >= 1.0:
                score = 0.999
            score   = round(score, 3)
            success = score > 0.5

        except Exception as e:
            print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
            score = 0.001

        finally:
            rewards_str = ",".join(f"{r:.4f}" for r in rewards)
            print(
                f"[END] task={task_id} success={str(success).lower()} "
                f"steps={steps} score={score:.3f} rewards={rewards_str}",
                flush=True,
            )


if __name__ == "__main__":
    main()