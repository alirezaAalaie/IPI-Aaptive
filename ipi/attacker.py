"""
BaseAttacker — the ABC every IPI attack implements, and the machinery it shares.

The seam is one method::

    single_attack(target, instance, verbose) -> AttackDataset

``instance`` is the *scenario*; the returned dataset holds the candidate(s) the attack
wants to report, with the injection on ``jailbreak_prompt``, the victim's reply in
``target_responses`` and the judge's verdict in ``eval_results``. Nothing else. The
``ScenarioResult`` that ``AttackEvaluator`` consumes is built from that dataset here,
once, rather than by hand in every recipe.

    BaseAttacker.run_scenario()   built-in: single_attack → best_of → build_result
    BaseAttacker.report()         stamp the bookkeeping and hand back the dataset
    BaseAttacker.update() / log() the running counters, as upstream's AttackerBase has

Why this exists
---------------
Each recipe used to carry a ``*Result`` dataclass, a ``run_*`` free function taking a
``target_fn``, and a ``run_scenario`` adapter mapping one to the other — 1091 lines
across thirteen modules, all of it plumbing, none of it algorithm. One carrier
(``Instance``) does the same job, which is what EasyJailbreak's recipes do and why
theirs are half the size.

What was deliberately **not** copied from upstream
--------------------------------------------------
  * **The target and the dataset stay out of ``__init__``.** Upstream's ``AttackerBase``
    takes ``target_model`` and ``jailbreak_datasets`` as constructor arguments, so
    sweeping six defenses across thirteen attacks means seventy-eight attacker objects.
    Ours takes them per call — that is what makes the baseline sweep a nested loop.
  * **The counters are not ``sum(eval_results)``.** Upstream's ``update()`` adds
    ``Instance.num_jailbreak``, which sums *scores* when the evaluator is a 1-10
    ``*GetScore`` — TAP's is. Ours count the adjudicated ``ScenarioResult.success``, so
    ``current_jailbreak`` means the same thing whatever evaluator the recipe used.
    They are progress bookkeeping regardless: the reported ASR comes from
    ``AttackEvaluator``, never from here.

Migration state
---------------
``run_scenario`` is concrete and ``single_attack`` is the method to write. A recipe that
has not been converted yet overrides ``run_scenario`` directly and is untouched by this;
a converted one implements ``single_attack`` and inherits the rest. ``__init__`` rejects
a subclass that defines neither, which is the guard the old ``@abstractmethod`` gave.

Hierarchy
---------
    BaseAttacker (ABC)
    ├── StaticAttacker      one query, no judge needed
    ├── AdaptiveAttacker    iterative search
    └── JudgeGuidedAttacker  iterative, and the judge is mandatory (TAP, PAIR)
"""
from __future__ import annotations

from abc import ABC
from typing import Any, Optional

from .config import (
    ATTACK_MAX_TOKENS, ATTACK_TEMP, ATTACK_TOP_P, SUCCESS_THRESHOLD,
)
from .datasets import AttackDataset, Instance
from .metrics import Evaluator, ScenarioResult
from .victim import Victim

__all__ = [
    "BaseAttacker", "StaticAttacker", "AdaptiveAttacker", "JudgeGuidedAttacker",
    "as_attacker_llm", "REPORT_KEY", "INJECTION_KEY",
]

def as_attacker_llm(model):
    """
    A model *string* becomes an LLM at the house attacker settings — ``APILLM``, or
    ``KaggleLLM`` for a ``kaggle/`` prefix; anything else passes through untouched.

    Every recipe that drives an attacker LLM did this inline with the same three
    ``config`` constants. The test is on ``str`` and not on ``UnifiedLLM`` deliberately:
    the mutation operators only need ``callable(str) -> str``, so a bare function or a
    stub is a legitimate attacker model and must not be fed to the model-name parsing.
    ``litellm`` is imported lazily, so ``import ipi.attacker`` stays cheap.
    """
    if not isinstance(model, str):
        return model
    from .llm_unified import make_llm
    return make_llm(model, temperature=ATTACK_TEMP, top_p=ATTACK_TOP_P,
                    max_tokens=ATTACK_MAX_TOKENS)


#: ``attack_attrs`` slot holding a candidate's report bookkeeping (see ``report``).
REPORT_KEY = "_report"

#: ``attack_attrs`` slot holding the string actually sent to the victim. Written by
#: ``harness.VictimQuery``; differs from ``jailbreak_prompt`` whenever that field is a
#: template with a ``{query}`` placeholder still in it.
INJECTION_KEY = "injection"


class BaseAttacker(ABC):
    """
    Abstract base class for all IPI attackers. See the module docstring for the seam.

    Args:
        judge: Guidance ``Evaluator`` (``ipi.metrics``), owned by this attacker. It
               steers the search only — ``AttackEvaluator`` recomputes success against
               the scenario's ground truth and overwrites whatever was concluded here.
    """

    #: Attack label in the results table. Defaults to the class name minus "Attacker".
    _ATTACK_NAME: Optional[str] = None

    def __init__(self, judge: Optional[Evaluator] = None):
        cls = type(self)
        if (cls.run_scenario is BaseAttacker.run_scenario
                and cls.single_attack is BaseAttacker.single_attack):
            raise TypeError(
                f"{cls.__name__} implements neither single_attack() nor "
                "run_scenario(); one of the two is the attack.")
        self.judge = judge
        # Progress counters, in upstream's names. Reported by log(); not the ASR.
        self.current_query = 0
        self.current_jailbreak = 0
        self.current_reject = 0
        self.current_iteration = 0

    # ------------------------------------------------------------------
    # Identity / capability
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The attack's label — matches what ``AttackEvaluator`` derives."""
        return self._ATTACK_NAME or type(self).__name__.replace("Attacker", "").lower()

    @classmethod
    def requires_local_target(cls) -> bool:
        """True if this attack needs a LocalLLM victim (white-box: GCG, AutoDAN, BEAST)."""
        return False

    @property
    def is_iterative(self) -> bool:
        """True if the attack performs iterative optimisation/search."""
        return False

    # ------------------------------------------------------------------
    # The seam
    # ------------------------------------------------------------------

    def single_attack(
        self,
        target: Victim,
        instance: Instance,
        verbose: bool = False,
    ) -> AttackDataset:
        """
        Run the attack against one scenario and return the candidate(s) to report.

        Args:
            target:   Victim — TargetLLM, or any defense wrapping one.
            instance: The scenario. ``instance.query`` is the attacker's goal; the
                      IPI context (user_task, tool_schema, pipeline_context,
                      target_str, optimization_target) is in ``attack_attrs`` and is
                      read through ``ipi.harness`` / ``metrics.resolve_attack_target``,
                      never by hand — ``attack_attrs.get`` returns ``None`` on a typo
                      where an attribute access would have raised.
            verbose:  Enable INFO-level logging.

        Returns:
            An ``AttackDataset``, normally the single best candidate, built with
            ``self.report(best, query, **extra)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not been converted to the single_attack seam.")

    def run_scenario(
        self,
        target: Victim,
        instance: Instance,
        verbose: bool = False,
    ) -> ScenarioResult:
        """
        Run one scenario and adapt the result for ``AttackEvaluator``.

        Concrete on purpose: an unconverted recipe overrides this and never calls
        ``single_attack``. The ``success`` on the returned result is the attack's *own*
        verdict, used for its early stopping — ``AttackEvaluator`` overwrites it, so
        never report it.
        """
        dataset = self.single_attack(target, instance, verbose=verbose)
        result = self.build_result(instance, dataset)
        self.update(result)
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def report(
        best: Instance,
        query: Any = 0,
        *,
        success: Optional[bool] = None,
        score: Optional[int] = None,
        **extra,
    ) -> AttackDataset:
        """
        Stamp a candidate with the per-scenario bookkeeping and wrap it for return.

        The last line of a ``single_attack``::

            return self.report(best, query, depth_reached=depth)

        Args:
            best:    The candidate to report.
            query:   The ``harness.VictimQuery`` the recipe queried through (its
                     ``n_queries`` is read), or a plain int.
            success: The attack's own verdict. ``None`` derives it from the score.
            score:   The 1-10 guidance score. ``None`` reads ``eval_results[-1]``.
            **extra: Recipe-specific fields for ``ScenarioResult.extra`` — the only
                     part of the result that legitimately varies per attack.
        """
        n_queries = getattr(query, "n_queries", query)
        best.attack_attrs[REPORT_KEY] = {
            "n_queries": int(n_queries or 0),
            "success": success,
            "score": score,
            "extra": extra,
        }
        return AttackDataset([best])

    @staticmethod
    def score_of(instance: Instance) -> int:
        """A candidate's 1-10 score — its stamped score, else ``eval_results[-1]``."""
        stamped = instance.attack_attrs.get(REPORT_KEY, {}).get("score")
        if stamped is not None:
            return Evaluator.as_score(stamped)
        if instance.eval_results:
            return Evaluator.as_score(instance.eval_results[-1])
        return 1

    @classmethod
    def best_of(cls, dataset: AttackDataset) -> Optional[Instance]:
        """
        The candidate to report: a stamped success first, then the highest score.

        Ties go to the earliest, which for every recipe here is the one found first.
        """
        best: Optional[Instance] = None
        best_key: Optional[tuple] = None
        for instance in dataset:
            stamped = instance.attack_attrs.get(REPORT_KEY, {}).get("success")
            key = (bool(stamped), cls.score_of(instance))
            if best_key is None or key > best_key:
                best, best_key = instance, key
        return best

    @staticmethod
    def normalise_scores(dataset: AttackDataset) -> AttackDataset:
        """
        Collapse each candidate's ``eval_results`` to a single 1-10 int.

        A recipe's ``judge=`` slot takes either kind of evaluator: a ``*GetScore``
        already wrote an int, a ``*Judge`` wrote a bool. Anything that reads
        ``eval_results[-1]`` as a magnitude — ``SelectBasedOnScores`` does — needs them
        to arrive as the same type, and ``Evaluator.as_score`` is the one mapping
        (True -> 10, False -> 1) so a threshold means the same thing for both.
        """
        for candidate in dataset:
            candidate.eval_results = [
                Evaluator.as_score(candidate.eval_results[-1])
                if candidate.eval_results else 1
            ]
        return dataset

    @classmethod
    def keep_best(
        cls,
        best: Optional[Instance],
        dataset: AttackDataset,
    ) -> Optional[Instance]:
        """
        The better of ``best`` and the best candidate in ``dataset``.

        The one line that replaces the ``best_score`` / ``best_injection`` /
        ``best_response`` triple every search recipe carried — sixty references across
        the package, each its own chance to update two of the three. Keeping the
        winning ``Instance`` rather than three parallel scalars also means the reported
        row is a real node of the search tree, with its lineage intact.

        Global, not per level: upstream's TAP takes ``max()`` over the *survivors of the
        last level*, which silently discards a better candidate found at depth 2 when
        depth 7 scores worse.
        """
        challenger = cls.best_of(dataset)
        if challenger is None:
            return best
        if best is None:
            return challenger
        return challenger if cls.score_of(challenger) > cls.score_of(best) else best

    def build_result(self, scenario: Instance, dataset: AttackDataset) -> ScenarioResult:
        """Turn what ``single_attack`` returned into the row ``AttackEvaluator`` wants."""
        best = self.best_of(dataset)
        if best is None:
            return ScenarioResult(
                scenario_id=scenario.id, goal=scenario.query or "", success=False,
                score=0, injection="", target_response="", n_queries=0,
                attack=self.name, extra={"note": "the attack produced no candidate"},
            )

        stamped = best.attack_attrs.get(REPORT_KEY, {})
        score = self.score_of(best)
        success = stamped.get("success")
        if success is None:
            success = (self.judge.is_success(score) if self.judge is not None
                       else score >= SUCCESS_THRESHOLD)
        # The string the victim actually saw. VictimQuery writes it; it differs from
        # jailbreak_prompt whenever that field is still a {query} template.
        injection = best.attack_attrs.get(INJECTION_KEY) or best.jailbreak_prompt or ""

        return ScenarioResult(
            scenario_id=scenario.id,
            goal=scenario.query or "",
            success=bool(success),
            score=score,
            injection=injection,
            target_response=best.target_responses[-1] if best.target_responses else "",
            n_queries=int(stamped.get("n_queries", 0)),
            attack=self.name,
            extra=dict(stamped.get("extra", {})),
        )

    # ------------------------------------------------------------------
    # Running counters (upstream's AttackerBase.update / log)
    # ------------------------------------------------------------------

    def update(self, result: ScenarioResult):
        """Fold one scenario's outcome into the running counters."""
        self.current_iteration += 1
        self.current_query += result.n_queries
        self.current_jailbreak += int(bool(result.success))
        self.current_reject += int(not result.success)

    def log(self):
        """Print the running counters. Progress only — the ASR is AttackEvaluator's."""
        import logging
        log = logging.getLogger(__name__)
        log.info("======%s report:======", self.name)
        log.info("Scenarios:      %d", self.current_iteration)
        log.info("Victim queries: %d", self.current_query)
        log.info("Jailbroken:     %d", self.current_jailbreak)
        log.info("Rejected:       %d", self.current_reject)
        log.info("========Report End===========")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(judge={self.judge!r})"


class StaticAttacker(BaseAttacker):
    """
    Single-query, template-based prompt injection (the OPI one-shots, DeepInception, ICA).

    The injection is built deterministically from a fixed template, so no judge is
    needed during generation — ``judge`` only annotates.
    """

    @property
    def is_iterative(self) -> bool:
        return False


class AdaptiveAttacker(BaseAttacker):
    """
    Iterative, search-based or gradient-based attacks.

    The injection is optimised across many victim queries using an LLM judge, logprob
    signals, target loss or white-box gradients.
    """

    @property
    def is_iterative(self) -> bool:
        return True


class JudgeGuidedAttacker(AdaptiveAttacker):
    """
    Adaptive attacks that need a guidance signal at *every* step (TAP, PAIR).

    Both score every candidate to decide which branch to expand, so a ``*GetScore``
    evaluator is required rather than optional. That score steers the search only.
    """

    def __init__(self, judge: Evaluator):
        if judge is None:
            raise ValueError(
                f"{type(self).__name__} requires a guidance Evaluator "
                f"(e.g. metrics.EvaluatorIPIGetScore).")
        super().__init__(judge=judge)
