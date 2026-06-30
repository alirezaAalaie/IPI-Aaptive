"""
BaseAttacker — abstract base class for all IPI attack implementations.

Every attack (TAP, PAIR, RS, Beam-RS, BEAST) subclasses BaseAttacker and:
  1. Takes a judge as a required constructor argument (owns its judge).
  2. Sets all attack hyperparameters at construction time.
  3. Implements run_scenario(target, scenario, verbose) → ScenarioResult.

Usage
-----
    from ipi.tap import TAPAttacker
    from ipi.judges import IPILLMJudge
    from ipi.llm_unified import APILLM
    from ipi.evaluator import AttackEvaluator

    judge    = IPILLMJudge(model="gpt-4o-mini")
    attacker = TAPAttacker(judge=judge, attacker_llm=APILLM("gpt-4o"), depth=10)
    evaluator = AttackEvaluator(target=APILLM("gpt-4o-mini", system_prompt="..."),
                                attacker=attacker)
    result = evaluator.run(dataset)

Hierarchy
---------
    BaseAttacker (ABC)
    ├── TAPAttacker      (tap.py)
    ├── PAIRAttacker     (pair.py)
    ├── RSAttacker       (adaptive.py)
    ├── BeamRSAttacker   (adaptive.py)
    └── BEASTAttacker    (beast.py)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .dataset import IPIScenario
from .judges import Judge

if TYPE_CHECKING:
    from .evaluator import ScenarioResult
    from .victim import Victim


class BaseAttacker(ABC):
    """
    Abstract base class for IPI attackers.

    Args:
        judge: Judge instance. Owned exclusively by this attacker —
               it is never passed to AttackEvaluator or shared externally.
    """

    def __init__(self, judge: Judge):
        self.judge = judge

    @classmethod
    def requires_local_target(cls) -> bool:
        """
        Return True if this attack requires a LocalLLM target model.
        Default False; overridden by BEASTAttacker.
        """
        return False

    @abstractmethod
    def run_scenario(
        self,
        target: "Victim",
        scenario: IPIScenario,
        verbose: bool = False,
    ) -> "ScenarioResult":
        """
        Run the attack on a single IPI scenario.

        Args:
            target:   Victim instance (TargetLLM or custom defense subclass).
            scenario: The IPI scenario with goal, user_task, tool_schema, etc.
            verbose:  Enable INFO-level logging.

        Returns:
            ScenarioResult with success flag, score, injection string, and metadata.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(judge={self.judge!r})"
