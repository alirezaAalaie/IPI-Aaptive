"""
Evaluator — the ABC every metric in this package implements.

Mirrors ``easyjailbreak.metrics.Evaluator.Evaluator``: an evaluator takes an
``AttackDataset``, walks it, and writes verdicts onto each ``Instance``. Upstream draws
no line between "guidance" and "success" — both live in ``metrics/``, told apart only by
what they write:

    *GetScore  →  an int 1-10 in ``instance.eval_results``  — steers a search
    *Judge / *Match →  a bool in ``instance.eval_results``  — decides ASR

Keep that distinction in mind when pairing one with a selector: ``Instance.num_jailbreak``
is ``sum(eval_results)``, so a reward-driven selector (UCB, MCTS) needs a **binary**
evaluator; a 1-10 one belongs with ``SelectBasedOnScores``.

The scalar path
---------------
Recipes have not migrated to the carrier yet (Phase H) — they hold plain
``(injection, response)`` strings and call ``judge.score(...)``. ``Evaluator.score()``
is the adapter: it packs those scalars into a throwaway ``Instance``, runs the same
``_evaluate`` the dataset path runs, and unpacks the verdict. There is no second
implementation of anything. When the recipes take ``Instance`` objects the adapter
disappears, not the evaluator.

Deviations from upstream
------------------------
  * ``__call__`` returns the dataset (upstream returns ``None``) so evaluators chain
    like the mutation/constraint families do.
  * ``_evaluate`` **resets** ``instance.eval_results`` before writing. Upstream is
    inconsistent — ``EvalatorMatch`` and ``EvaluatorPatternJudge`` reset,
    ``EvaluatorPrefixExactMatch`` and ``EvaluatorGenerativeGetScore`` append — which
    makes ``num_jailbreak`` depend on how many times an instance has been evaluated.
    We reset everywhere.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import SUCCESS_THRESHOLD
from ..datasets import AttackDataset, Instance

__all__ = ["Evaluator", "instance_from_scalars", "get_attr"]


# Context keys accepted by the scalar path that name the same thing as an Instance
# field: ctx key -> where to look on the Instance.
_ATTR_ALIASES = {
    "target_tool_calls": ("target_str", "reference_responses"),
    "attacker_goal": ("query",),
}


def instance_from_scalars(injection: str = "", response: str = "", **ctx) -> Instance:
    """
    Pack a recipe's loose strings into an ``Instance`` for the scalar path.

    Mapping (the same one ``dual_verifiable.py`` uses, so an evaluator cannot tell
    which path it is on):

        attacker_goal / goal  -> query
        injection             -> jailbreak_prompt
        response              -> target_responses[-1]
        target_str / target_tool_calls -> reference_responses[-1]
        everything else       -> attack_attrs
    """
    goal = ctx.pop("attacker_goal", None) or ctx.pop("goal", None) or ""
    reference = ctx.get("target_str") or ctx.get("target_tool_calls") or ""
    return Instance(
        query=goal,
        jailbreak_prompt=injection,
        target_responses=[response],
        reference_responses=[reference] if reference else [],
        attack_attrs=dict(ctx),
    )


def get_attr(instance: Instance, name: str, default: str = "") -> str:
    """
    Read ``name`` off an Instance for prompt formatting.

    Looks at the instance's own fields first, then ``attack_attrs``, then the alias
    table. List-valued fields (``target_responses``, ``reference_responses``) yield
    their **last** element, matching upstream's ``_format``.
    """
    for key in (name, *_ATTR_ALIASES.get(name, ())):
        value = getattr(instance, key, None)
        if value is None:
            value = instance.attack_attrs.get(key)
        if isinstance(value, (list, tuple)):
            value = value[-1] if value else None
        if value not in (None, ""):
            return str(value)
    return default


class Evaluator(ABC):
    """
    Base class for every metric. See module docstring.

    Args:
        eval_model: The LLM an LLM-backed evaluator drives. ``None`` for the
                    deterministic ones.
    """

    #: Scalar-path threshold: ``score >= success_threshold`` counts as a hit. Only
    #: steers a recipe's early stop — never the reported ASR, which AttackEvaluator owns.
    success_threshold: int = SUCCESS_THRESHOLD

    def __init__(self, eval_model=None):
        self.eval_model = eval_model

    # ------------------------------------------------------------------
    # Dataset path (upstream seam)
    # ------------------------------------------------------------------

    def __call__(self, dataset: AttackDataset, **kwargs) -> AttackDataset:
        """Evaluate every instance in ``dataset`` in place, and return it."""
        for instance in dataset:
            self._evaluate(instance, **kwargs)
        return dataset

    @abstractmethod
    def _evaluate(self, instance: Instance, **kwargs):
        """Write this evaluator's verdict(s) onto ``instance.eval_results``."""

    # ------------------------------------------------------------------
    # Scalar path (used by the not-yet-migrated recipes)
    # ------------------------------------------------------------------

    def score(self, injection: str, response: str, **ctx) -> int:
        """
        Score one (injection, response) pair on 1-10.

        Args:
            injection: The injection string sent to the victim.
            response:  The victim's response.
            **ctx:     Optional IPI context — ``attacker_goal``, ``user_task``,
                       ``target_tool_calls``, ``actual_tools_called``.

        Returns:
            int in [1, 10]. A binary evaluator maps True->10, False->1, so it drops
            into a recipe's ``judge=`` slot unchanged.
        """
        instance = instance_from_scalars(injection, response, **ctx)
        self._evaluate(instance)
        if not instance.eval_results:
            return 1
        return self.as_score(instance.eval_results[-1])

    @staticmethod
    def as_score(result) -> int:
        """
        One ``eval_results`` entry as a 1-10 int.

        A recipe that prunes on score has to cope with either kind of evaluator in its
        ``judge=`` slot: a ``*GetScore`` already wrote an int, a ``*Judge`` wrote a bool.
        True maps to 10 and False to 1, so the same threshold works for both.
        """
        if isinstance(result, bool):
            return 10 if result else 1
        return max(1, min(10, int(result)))

    def is_success(self, score: int) -> bool:
        """Whether ``score`` clears this evaluator's threshold (early-stop only)."""
        return score >= self.success_threshold

    def __repr__(self) -> str:
        model = getattr(self.eval_model, "model_name", None) or getattr(
            self.eval_model, "model", None)
        suffix = f"(model={model!r})" if model else "()"
        return f"{type(self).__name__}{suffix}"
