"""
BaseAttacker — abstract base class for all IPI attack implementations.

Every attack (TAP, PAIR, RS, Beam-RS, BEAST) subclasses BaseAttacker and:
  1. Takes its guidance evaluator as a constructor argument (owns it).
  2. Sets all attack hyperparameters at construction time.
  3. Implements run_scenario(target, scenario, verbose) → ScenarioResult.

Usage
-----
    from ipi.tap import TAPAttacker
    from ipi.metrics import EvaluatorIPIGetScore
    from ipi.llm_unified import APILLM
    from ipi.metrics import AttackEvaluator

    judge    = EvaluatorIPIGetScore(model="gpt-4o-mini")
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
    ├── BEASTAttacker    (beast.py)
    ├── AutoDANAttacker  (autodan.py)   — local
    └── GCGAttacker      (gcg.py)       — local
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from .dataset import IPIScenario
from .metrics import Evaluator

if TYPE_CHECKING:
    from .metrics import ScenarioResult
    from .victim import Victim


class BaseAttacker(ABC):
    """
    Abstract base class for all IPI attackers.

    Subclasses implement ``run_scenario(target, scenario, verbose) -> ScenarioResult``.
    """

    def __init__(self, judge: Optional[Evaluator] = None):
        self.judge = judge

    @classmethod
    def requires_local_target(cls) -> bool:
        """
        Return True if this attack requires a LocalLLM target model.
        Default False; overridden by BEASTAttacker, GCGAttacker, etc.
        """
        return False

    @property
    @abstractmethod
    def is_iterative(self) -> bool:
        """Return True if the attack performs iterative optimization/search."""
        ...

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


class StaticAttacker(BaseAttacker):
    """
    Base class for single-query, template-based static prompt injection attacks (OPI).

    Static attacks (Naive, Escape, Ignore, FakeCompletion, Combined) construct
    the injection string deterministically from a fixed template in 1 query.
    They do NOT require an LLM judge during injection generation (judge is optional).

    Args:
        judge: Optional guidance Evaluator (used only if explicit quality scoring
               is requested). See ipi.metrics.
    """

    def __init__(self, judge: Optional[Evaluator] = None):
        super().__init__(judge=judge)

    @property
    def is_iterative(self) -> bool:
        return False


class AdaptiveAttacker(BaseAttacker):
    """
    Base class for iterative, search-based, or gradient-based adaptive attacks.

    Adaptive attacks optimize the injection iteratively using feedback from an
    LLM judge, logprob signals, target loss, or white-box gradients across
    multiple target model queries.

    Args:
        judge: Optional guidance Evaluator (ipi.metrics).
    """

    def __init__(self, judge: Optional[Evaluator] = None):
        super().__init__(judge=judge)

    @property
    def is_iterative(self) -> bool:
        return True


class JudgeGuidedAttacker(AdaptiveAttacker):
    """
    Base class for adaptive attacks that need a guidance signal at every step.

    TAP and PAIR score every candidate to decide which branch to expand, so a
    ``*GetScore`` evaluator (``ipi.metrics``) is required rather than optional. That
    score steers the search only — the reported ASR comes from ``AttackEvaluator``.

    Args:
        judge: Required guidance Evaluator.
    """

    def __init__(self, judge: Evaluator):
        if judge is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a guidance Evaluator "
                f"(e.g. metrics.EvaluatorIPIGetScore).")
        super().__init__(judge=judge)

