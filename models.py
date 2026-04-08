# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Chemical Discovery Env Environment.

The chemical_discovery_env environment is a chemical property / drug-likeness
prediction task with three difficulty modes (easy/medium/hard).
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Any, Dict, Literal, Optional


class ChemicalDiscoveryAction(Action):
    """Agent prediction for the current molecule."""

    action_type: Literal["predict"] = Field(
        default="predict", description="Action type (fixed)"
    )
    molecule_id: str = Field(..., description="Target molecule identifier")
    prediction: Dict[str, Any] = Field(
        ..., description="Task-specific prediction payload"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Model confidence in [0,1]"
    )


class ChemicalDiscoveryObservation(Observation):
    """Observation describing one molecule and its derived properties."""

    molecule_id: str = Field(..., description="Current molecule identifier")
    smiles: str = Field(..., description="SMILES chemical notation string")

    molecular_weight: float = Field(..., ge=0.0, le=5000.0)
    logp: float = Field(..., ge=-20.0, le=20.0)
    tpsa: float = Field(..., ge=0.0, le=1000.0)
    hbd: int = Field(..., ge=0, le=100)
    hba: int = Field(..., ge=0, le=200)
    rotatable_bonds: int = Field(..., ge=0, le=200)

    task_type: Literal["easy", "medium", "hard"] = Field(
        ..., description="Active task difficulty"
    )
    step_count: int = Field(..., ge=0, le=10_000, description="Step index in episode")

    # Optional convenience fields (safe for agents; graders remain deterministic)
    dataset_size: Optional[int] = Field(
        default=None, ge=1, description="Dataset size for the active task"
    )
