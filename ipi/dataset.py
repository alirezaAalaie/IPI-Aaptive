"""
LEGACY — the pre-carrier scenario type. Being replaced by ``ipi.datasets``.

``IPIScenario`` is an immutable input record with no room for search state (no
lineage, no visit counts, no per-candidate responses), which is why the attacks each
invented their own candidate type. ``ipi.datasets.Instance`` replaces it. This module
stays only until the recipes are migrated onto the carrier (refactor Phase H), then it
is deleted — see ``docs/ipi-refactor-plan.md``.

**Do not add anything here.** New code should use::

    from ipi.datasets import Instance, AttackDataset, DualVerifiableDataset

Removed from this module (the benchmark is DualVerifiable only):
  ManualIPIDataset · AgentDojoDataset · BipiaDataset (+ its target maps)
  · HijackDataset (+ HIJACK_ATTACKS)

IPIScenario fields
------------------
  id                  — unique identifier string
  user_task           — the agent's original legitimate task (what the user asked for)
  injection_goal      — the attacker's objective (what the injection should make the agent do)
  target_tool_calls   — expected tool call string if attack succeeds (for scorer context)
  tool_schema         — description of tools available to the agent (for attacker context)
  pipeline_context    — any additional context the agent would have (e.g. prior messages)
  metadata            — free-form dict for benchmarking metadata
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IPIScenario
# ---------------------------------------------------------------------------

@dataclass
class IPIScenario:
    """
    A single Indirect Prompt Injection evaluation scenario.

    Attributes:
        id:                  Unique scenario identifier (e.g. "agentdojo/workspace/s1").
        user_task:           The legitimate task the agent was asked to perform.
        injection_goal:      What the attacker wants the agent to do (attacker goal string).
        target_output:       Expected output string used for **evaluation / verification**.
                             Used by AttackEvaluator._check_success and BipiaSuccessEvaluator
                             to decide whether an attack succeeded.
                             May be a sentinel string (e.g. ``"__base64__"``); evaluators
                             resolve sentinels with rule-based checks.
        optimization_target: Short, literal string used as the **optimization target** by
                             gradient / search attacks (GCG, RS, Beam-RS, BEAST).
                             Must be a real token sequence — NO sentinel values here.
                             Attacks use it as the ``target_str`` for:
                               • RS/Beam-RS: ``get_target_token(optimization_target)``
                                 → first token whose logprob is maximized.
                               • GCG: ``optimization_target`` token ids form the CE loss target.
                             When empty, attacks fall back to ``target_output`` (legacy
                             behaviour, only safe when target_output is a real literal).
                             LLM-based attacks (TAP, PAIR) ignore this field entirely and
                             use their judge's scoring instead.
        tool_schema:         Human-readable description of available tools.
        pipeline_context:    Optional additional context the agent has at injection time.
        metadata:            Free-form metadata dict (suite name, task type, etc.).
    """
    id:                  str
    user_task:           str
    injection_goal:      str
    target_output:       str = ""
    optimization_target: str = ""
    tool_schema:         str = ""
    pipeline_context:    str = ""
    metadata:            dict = field(default_factory=dict)

    def __init__(
        self,
        id: str,
        user_task: str,
        injection_goal: str,
        target_output: str = "",
        optimization_target: str = "",
        tool_schema: str = "",
        pipeline_context: str = "",
        metadata: Optional[dict] = None,
        target_tool_calls: Optional[str] = None,
    ):
        self.id = id
        self.user_task = user_task
        self.injection_goal = injection_goal
        self.optimization_target = optimization_target
        self.tool_schema = tool_schema
        self.pipeline_context = pipeline_context
        self.metadata = metadata if metadata is not None else {}

        if target_tool_calls is not None:
            self.target_output = target_tool_calls
        else:
            self.target_output = target_output

    @property
    def target_tool_calls(self) -> str:
        """Legacy alias for backward compatibility."""
        return self.target_output

    @target_tool_calls.setter
    def target_tool_calls(self, value: str):
        self.target_output = value


    def to_attack_context(self) -> dict:
        """
        Build the ``context`` dict expected by run_tap / run_pair / run_attack.

        Returns:
            dict with keys: user_task, tool_schema, target_tool_calls, conversation_history
        """
        return {
            "user_task":          self.user_task,
            "tool_schema":        self.tool_schema,
            "target_tool_calls":  self.target_tool_calls,
            "conversation_history": self.pipeline_context,
        }

    def to_experiment_scenario(self) -> dict:
        """
        Build a scenario dict compatible with run_experiment's ``scenarios`` list.

        Returns:
            dict with keys: id, goal, context
        """
        return {
            "id":      self.id,
            "goal":    self.injection_goal,
            "context": self.to_attack_context(),
        }


# ---------------------------------------------------------------------------
# IPIDataset (abstract base)
# ---------------------------------------------------------------------------

class IPIDataset(ABC):
    """
    Abstract base class for IPI attack datasets.

    Subclasses implement __len__, __iter__, and __getitem__. All other methods
    (subset, to_list, to_experiment_scenarios) are provided by this base class.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of scenarios."""

    @abstractmethod
    def __iter__(self) -> Iterator[IPIScenario]:
        """Iterate over all IPIScenario objects."""

    @abstractmethod
    def __getitem__(self, idx: int) -> IPIScenario:
        """Return scenario at index idx."""

    def subset(self, n: int, seed: int = 42) -> "_ScenarioList":
        """
        Return a random subset of n scenarios.

        Args:
            n:    Number of scenarios to include (capped at len(self)).
            seed: Random seed for reproducibility.

        Returns:
            A list-backed IPIDataset with n randomly selected scenarios.
        """
        import random
        rng = random.Random(seed)
        items = list(self)
        selected = rng.sample(items, min(n, len(items)))
        return _ScenarioList(selected)

    def to_list(self) -> list[IPIScenario]:
        """Return all scenarios as a plain list."""
        return list(self)

    def to_experiment_scenarios(self) -> list[dict]:
        """
        Convert all scenarios to run_experiment-compatible dicts.

        Returns:
            List of dicts with keys: id, goal, context.
        """
        return [s.to_experiment_scenario() for s in self]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={len(self)})"


class _ScenarioList(IPIDataset):
    """
    Minimal list-backed IPIDataset — the return type of ``subset()``.

    Replaces the removed ``ManualIPIDataset``, which existed for the same purpose plus
    dict-parsing that nothing used.
    """

    def __init__(self, scenarios: list[IPIScenario]):
        self._scenarios = list(scenarios)

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self) -> Iterator[IPIScenario]:
        return iter(self._scenarios)

    def __getitem__(self, idx: int) -> IPIScenario:
        return self._scenarios[idx]


# ---------------------------------------------------------------------------
# DualVerifiableDataset
# ---------------------------------------------------------------------------

class DualVerifiableDataset(IPIDataset):
    """
    Dataset for dual-verifiable Indirect Prompt Injection (IPI) evaluation.

    Each scenario provides:
      1. Deterministic User Task ground-truth answer (user_target) for 0-LLM utility evaluation.
      2. Deterministic Attacker Target string (target_str) and explicit optimization_target
         token sequence for gradient/search attacks (GCG, RS, BEAST).

    Args:
        json_path: Path to the dual_verifiable_dataset.json file. If None, auto-locates it.
    """

    def __init__(self, json_path: Optional[str] = None):
        import os, json
        if json_path is None:
            ipi_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(ipi_dir, "data", "dual_verifiable_dataset.json")
            if os.path.exists(candidate):
                json_path = candidate
            else:
                # Try relative working directory
                json_path = "ipi/data/dual_verifiable_dataset.json"

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Dual-verifiable dataset JSON not found at {json_path}. "
                "Run `python3 scripts/build_dual_verifiable_dataset.py` to construct it."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._scenarios: list[IPIScenario] = []
        for s in data:
            scenario = IPIScenario(
                id=s["id"],
                user_task=s["user_task"],
                injection_goal=s["injection_goal"],
                target_output=s["target_str"],
                optimization_target=s["optimization_target"],
                tool_schema=s.get("tool_schema", ""),
                pipeline_context=s["clean_context"],
                metadata={
                    "user_target": s["user_target"],
                    "user_eval_mode": s.get("user_eval_mode", "contains"),
                    "attack_eval_mode": s.get("attack_eval_mode", "contains"),
                    "dataset_type": "dual_verifiable",
                    "domain": s.get("metadata", {}).get("domain", ""),
                    "attack_name": s.get("metadata", {}).get("attack_name", ""),
                    "clean_context": s["clean_context"],
                }
            )
            self._scenarios.append(scenario)

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self) -> Iterator[IPIScenario]:
        return iter(self._scenarios)

    def __getitem__(self, idx: int) -> IPIScenario:
        return self._scenarios[idx]


