# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chemical Discovery Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import ChemicalDiscoveryAction, ChemicalDiscoveryObservation


class ChemicalDiscoveryEnv(
    EnvClient[ChemicalDiscoveryAction, ChemicalDiscoveryObservation, State]
):
    """
    Client for the Chemical Discovery Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ChemicalDiscoveryEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(ChemicalDiscoveryAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ChemicalDiscoveryEnv.from_docker_image("chemical_discovery_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ChemicalDiscoveryAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ChemicalDiscoveryAction) -> Dict:
        """
        Convert ChemicalDiscoveryAction to JSON payload for step message.

        Args:
            action: ChemicalDiscoveryAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "molecule_id": action.molecule_id,
            "prediction": action.prediction,
            "confidence": action.confidence,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ChemicalDiscoveryObservation]:
        """
        Parse server response into StepResult[ChemicalDiscoveryObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ChemicalDiscoveryObservation
        """
        obs_data = payload.get("observation", {})
        observation = ChemicalDiscoveryObservation(
            molecule_id=obs_data.get("molecule_id", ""),
            smiles=obs_data.get("smiles", ""),
            molecular_weight=obs_data.get("molecular_weight", 0.0),
            logp=obs_data.get("logp", 0.0),
            tpsa=obs_data.get("tpsa", 0.0),
            hbd=obs_data.get("hbd", 0),
            hba=obs_data.get("hba", 0),
            rotatable_bonds=obs_data.get("rotatable_bonds", 0),
            task_type=obs_data.get("task_type", "easy"),
            step_count=obs_data.get("step_count", 0),
            dataset_size=obs_data.get("dataset_size"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        # State allows extra fields; preserve them if present.
        return State(**payload)
