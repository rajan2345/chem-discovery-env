# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Chemical Discovery Env Environment Implementation.

Chemical discovery environment with three deterministic tasks:
- easy: binary drug-likeness classification (Lipinski rule-of-5)
- medium: count Lipinski violations
- hard: predict multi-objective scores (activity/safety/synthesizability)
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ChemicalDiscoveryAction, ChemicalDiscoveryObservation
except ImportError:
    from models import ChemicalDiscoveryAction, ChemicalDiscoveryObservation

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


@dataclass(frozen=True)
class _Mol:
    mol_id: str
    smiles: str
    ground_truth: dict
    props: dict


class ChemicalDiscoveryEnvironment(Environment):
    """
    Chemical Discovery environment.

    Episodes are fixed-length (30 steps). Each step presents a molecule and the
    agent submits a task-specific prediction. Reward is computed deterministically
    from prediction vs ground-truth, with partial credit for medium/hard tasks.
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=None, step_count=0)
        self._task: str = os.getenv("CHEM_TASK", "easy")
        self._max_steps: int = int(os.getenv("CHEM_MAX_STEPS", "30"))
        self._episode_steps: int = 0
        self._cumulative_reward: float = 0.0

        self._molecules: list[_Mol] = []
        self._idx: int = 0
        self._loaded_task: str | None = None
        self._data_dir = Path(os.getenv("CHEM_DATA_DIR", "data"))
        self._rng = random.Random()

    def _load_task(self, task: str) -> None:
        if self._loaded_task == task and self._molecules:
            return
        path = self._data_dir / f"{task}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._molecules = [
            _Mol(m["id"], m["smiles"], m["ground_truth"], self._calc_props(m["smiles"]))
            for m in raw
        ]
        self._loaded_task = task
        self._idx = 0

    def _calc_props(self, smiles: str) -> dict:
        # RDKit is preferred. If unavailable (e.g. local dev), fall back to
        # conservative placeholder values so the env still runs; docker will
        # include RDKit and produce accurate values.
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import Descriptors  # type: ignore

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            return {
                "molecular_weight": float(Descriptors.MolWt(mol)),
                "logp": float(Descriptors.MolLogP(mol)),
                "tpsa": float(Descriptors.TPSA(mol)),
                "hbd": int(Descriptors.NumHDonors(mol)),
                "hba": int(Descriptors.NumHAcceptors(mol)),
                "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
            }
        except Exception:
            # Very rough, deterministic placeholders derived from string length.
            l = max(1, len(smiles))
            return {
                "molecular_weight": float(min(5000.0, 10.0 * l)),
                "logp": float(min(20.0, max(-20.0, (l % 40) / 4.0 - 5.0))),
                "tpsa": float(min(1000.0, (l % 200) * 1.5)),
                "hbd": int(l % 6),
                "hba": int(l % 11),
                "rotatable_bonds": int(l % 15),
            }

    def _grade(self, task: str, pred: dict, gt: dict) -> float:
        try:
            if task == "easy":
                return (
                    1.0
                    if bool(pred["is_drug_like"]) == bool(gt["is_drug_like"])
                    else 0.0
                )
            if task == "medium":
                err = abs(int(pred["violations"]) - int(gt["violations"]))
                return max(0.0, 1.0 - 0.2 * err)
            if task == "hard":
                a = max(0.0, 1.0 - abs(float(pred["activity"]) - float(gt["activity"])))
                s = max(0.0, 1.0 - abs(float(pred["safety"]) - float(gt["safety"])))
                y = max(
                    0.0,
                    1.0
                    - abs(
                        float(pred["synthesizability"]) - float(gt["synthesizability"])
                    ),
                )
                return 0.4 * a + 0.3 * s + 0.3 * y
        except Exception:
            return 0.0
        return 0.0

    def _obs(self, mol: _Mol) -> ChemicalDiscoveryObservation:
        p = mol.props
        return ChemicalDiscoveryObservation(
            molecule_id=mol.mol_id,
            smiles=mol.smiles,
            molecular_weight=float(p["molecular_weight"]),
            logp=float(p["logp"]),
            tpsa=float(p["tpsa"]),
            hbd=int(p["hbd"]),
            hba=int(p["hba"]),
            rotatable_bonds=int(p["rotatable_bonds"]),
            task_type=self._task,  # type: ignore[arg-type]
            step_count=self._episode_steps,
            dataset_size=len(self._molecules) if self._molecules else None,
            done=False,
            reward=0.0,
            metadata={},
        )

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, task: str = "easy"
    ) -> ChemicalDiscoveryObservation:  # type: ignore[override]
        """
        Reset the environment.

        Returns:
            ChemicalDiscoveryObservation for the first molecule
        """
        if task not in {"easy", "medium", "hard"}:
            task = "easy"
        self._task = task
        self._load_task(task)

        if seed is not None:
            self._rng = random.Random(int(seed))
            self._rng.shuffle(self._molecules)

        self._episode_steps = 0
        self._cumulative_reward = 0.0
        self._idx = 0

        eid = episode_id or f"ep_{int(time.time() * 1000)}_{uuid4().hex[:8]}"
        self._state = State(episode_id=eid, step_count=0)

        first = self._molecules[self._idx]
        obs = self._obs(first)
        obs.reward = 0.0
        obs.metadata = {"episode_id": eid}
        return obs

    def step(self, action: ChemicalDiscoveryAction) -> ChemicalDiscoveryObservation:  # type: ignore[override]
        """
        Execute a step by grading the agent's prediction.

        Args:
            action: ChemicalDiscoveryAction containing the prediction

        Returns:
            Next-molecule observation with reward from the action
        """
        # Ensure we have molecules loaded even if someone calls step() first.
        if self._loaded_task is None or not self._molecules:
            self.reset(task=self._task)

        current = self._molecules[self._idx]
        reward = float(self._grade(self._task, action.prediction, current.ground_truth))
        reward = max(0.0, min(1.0, reward))

        self._episode_steps += 1
        self._cumulative_reward += reward
        self._state.step_count = self._episode_steps

        done = self._episode_steps >= self._max_steps
        if not done:
            self._idx = (self._idx + 1) % len(self._molecules)
        nxt = self._molecules[self._idx]

        obs = self._obs(nxt)
        obs.reward = reward
        obs.done = done
        obs.metadata = {
            "episode_id": self._state.episode_id,
            "task": self._task,
            "current_step": self._episode_steps,
            "cumulative_reward": self._cumulative_reward,
            "graded_molecule_id": current.mol_id,
        }
        return obs

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        # State supports extra fields; include useful metadata.
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task=self._task,
            cumulative_reward=self._cumulative_reward,
            current_index=self._idx,
            dataset_size=len(self._molecules) if self._molecules else 0,
        )
