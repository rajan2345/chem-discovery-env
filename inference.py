"""
Baseline Inference Script
========================
MANDATORY - Uses judges' injected API_BASE_URL, API_KEY, and MODEL_NAME.
"""

import os
import asyncio
import json
import textwrap
from typing import List, Optional, Dict, Any

from openai import OpenAI

from client import ChemicalDiscoveryEnv
from models import ChemicalDiscoveryAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or ""
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

BENCHMARK = os.getenv("BENCHMARK", "chem-discovery-env")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

MAX_TOTAL_REWARD = float(MAX_STEPS)


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
    return {"activity": 0.5, "safety": 0.5, "synthesizability": 0.5}


def _system_prompt(task: str) -> str:
    if task == "easy":
        return 'You are predicting drug-likeness. Output ONLY JSON: {"is_drug_like": true|false}'
    if task == "medium":
        return (
            'You are predicting violation count. Output ONLY JSON: {"violations": 0-4}'
        )
    return 'Predict multi-objective scores. Output ONLY JSON: {"activity":0-1,"safety":0-1,"synthesizability":0-1}'


def _user_prompt(task: str, obs: Dict[str, Any]) -> str:
    return f"Task: {task}\nMolecule: {obs.get('molecule_id')}\nProperties: MW={obs.get('molecular_weight')}, LogP={obs.get('logp')}, HBD={obs.get('hbd')}, HBA={obs.get('hba')}\nPredict now. Return ONLY JSON."


async def _run_task_async(task: str, client: Optional[OpenAI]) -> None:
    env = ChemicalDiscoveryEnv(base_url=ENV_BASE_URL)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    last_error: Optional[str] = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task)
        obs = result.observation
        if hasattr(obs, "model_dump"):
            obs = obs.model_dump()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            try:
                if client:
                    print(f"[DEBUG] Making LLM call for {task} step {step}", flush=True)
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
                    try:
                        pred = json.loads(text)
                    except Exception:
                        start = text.find("{")
                        end = text.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            pred = json.loads(text[start : end + 1])
                        else:
                            pred = _heuristic_prediction(task, obs)
                    last_error = None
                else:
                    pred = _heuristic_prediction(task, obs)
                    last_error = "no_api_key"
            except Exception as exc:
                pred = _heuristic_prediction(task, obs)
                last_error = str(exc)

            action_obj = ChemicalDiscoveryAction(
                molecule_id=obs.get("molecule_id", ""),
                prediction=pred,
                confidence=0.7,
            )
            if hasattr(action_obj, "model_dump"):
                action_dict = action_obj.model_dump(exclude_none=True)
            else:
                action_dict = action_obj
            action_str = _compact_json(action_dict)

            result = await env.step(action_obj)
            obs = result.observation
            if hasattr(obs, "model_dump"):
                obs = obs.model_dump()

            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step, action=action_str, reward=reward, done=done, error=last_error
            )

            if done:
                break

        score = (sum(rewards) / MAX_TOTAL_REWARD) if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def _main_async() -> None:
    client = None
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in ("easy", "medium", "hard"):
        await _run_task_async(task, client)


def main() -> None:
    print(f"[DEBUG] API_KEY set: {bool(API_KEY)}", flush=True)
    print(f"[DEBUG] API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME: {MODEL_NAME}", flush=True)
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
