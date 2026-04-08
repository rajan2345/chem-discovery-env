"""
Baseline Inference Script Example
================================
MANDATORY (per hackathon.md + sample_inference.py):
- Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN (or API_KEY fallback), IMAGE_NAME (optional)
- Uses OpenAI client for all LLM calls
- Emits structured stdout logs in [START] / [STEP] / [END] format

This baseline runs the environment for each task: easy, medium, hard.
It uses an LLM when credentials are provided; otherwise it falls back to a
deterministic heuristic so you can still sanity-check locally.
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import ChemicalDiscoveryEnv
from models import ChemicalDiscoveryAction

IMAGE_NAME = os.getenv("IMAGE_NAME")  # If you are using docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = os.getenv("BENCHMARK", "chem-discovery-env")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

MAX_TOTAL_REWARD = float(MAX_STEPS)  # reward per step is in [0,1]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _heuristic_prediction(task: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    # Rule-of-5 heuristics using already-provided properties.
    if task == "easy":
        is_drug_like = (
            obs["molecular_weight"] < 500
            and obs["logp"] < 5
            and obs["hbd"] <= 5
            and obs["hba"] <= 10
        )
        return {"is_drug_like": bool(is_drug_like)}
    if task == "medium":
        violations = 0
        if obs["molecular_weight"] >= 500:
            violations += 1
        if obs["logp"] >= 5:
            violations += 1
        if obs["hbd"] > 5:
            violations += 1
        if obs["hba"] > 10:
            violations += 1
        return {"violations": int(violations)}
    # hard: neutral baseline
    return {"activity": 0.5, "safety": 0.5, "synthesizability": 0.5}


def _system_prompt(task: str) -> str:
    if task == "easy":
        return textwrap.dedent(
            """
            You are predicting drug-likeness using Lipinski-style heuristics.
            Output ONLY a JSON object with key: is_drug_like (boolean).
            """
        ).strip()
    if task == "medium":
        return textwrap.dedent(
            """
            You are predicting Lipinski rule-of-5 violation count.
            Output ONLY a JSON object with key: violations (integer).
            """
        ).strip()
    return textwrap.dedent(
        """
        You are predicting multi-objective scores.
        Output ONLY a JSON object with keys:
          activity (float 0..1), safety (float 0..1), synthesizability (float 0..1)
        """
    ).strip()


def _user_prompt(task: str, obs: Dict[str, Any]) -> str:
    # Keep prompt compact and stable.
    payload = {
        "task": task,
        "molecule_id": obs.get("molecule_id"),
        "smiles": obs.get("smiles"),
        "properties": {
            "molecular_weight": obs.get("molecular_weight"),
            "logp": obs.get("logp"),
            "tpsa": obs.get("tpsa"),
            "hbd": obs.get("hbd"),
            "hba": obs.get("hba"),
            "rotatable_bonds": obs.get("rotatable_bonds"),
        },
    }
    return "Predict now. Return only JSON.\n" + _compact_json(payload)


def _llm_prediction(client: OpenAI, task: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": _system_prompt(task)},
            {"role": "user", "content": _user_prompt(task, obs)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()
    # Robust parse: allow model to return surrounding text.
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


async def run_task(task: str) -> None:
    client = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception:
            pass

    # Prefer docker image if provided; else connect to a running server.
    env = None
    try:
        env = (
            ChemicalDiscoveryEnv.from_docker_image(IMAGE_NAME)
            if IMAGE_NAME
            else ChemicalDiscoveryEnv(base_url=ENV_BASE_URL)
        )
    except Exception as e:
        log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
        print(f"[ERROR] Failed to connect to environment: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_action_error: Optional[str] = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task)
        obs = (
            result.observation.model_dump()
            if hasattr(result.observation, "model_dump")
            else dict(result.observation)
        )

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            try:
                if client is not None:
                    try:
                        pred = _llm_prediction(client, task, obs)
                        last_action_error = None
                    except Exception as llm_err:
                        pred = _heuristic_prediction(task, obs)
                        last_action_error = str(llm_err)
                else:
                    pred = _heuristic_prediction(task, obs)
                    last_action_error = None
            except Exception as exc:
                pred = _heuristic_prediction(task, obs)
                last_action_error = str(exc)

            action_obj = ChemicalDiscoveryAction(
                molecule_id=obs.get("molecule_id", ""),
                prediction=pred,
                confidence=0.7 if client is not None else 0.9,
            )
            action_str = _compact_json(action_obj.model_dump(exclude_none=True))

            result = await env.step(action_obj)
            obs = (
                result.observation.model_dump()
                if hasattr(result.observation, "model_dump")
                else dict(result.observation)
            )

            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=last_action_error,
            )

            if done:
                break

        score = (sum(rewards) / MAX_TOTAL_REWARD) if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Task {task} failed: {e}", flush=True)
    finally:
        if env:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    for task in ("easy", "medium", "hard"):
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())
