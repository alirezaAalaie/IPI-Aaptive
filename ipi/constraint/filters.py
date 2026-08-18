"""
LLM-backed constraints — drop candidates a model says are not worth a query.

    DeleteOffTopic   TAP's Phase-1 pruning: does this prompt still ask for the goal?
    DeleteHarmLess   ReNeLLM's filter: is the rewritten payload still harmful?

Both prompts are verbatim from upstream and live in the seed registry under
``constraint.<Name>.original``, checked by ``scripts/check_seed_fidelity.py``.

On ``DeleteHarmLess`` and IPI
-----------------------------
ReNeLLM runs this between rewrites so its six meaning-preserving operators cannot quietly
sand the harm off the payload. Our ``attacks/renellm.py`` **drops it on purpose**: an IPI
payload is an *instruction* ("Print only HACKED"), not harmful content, so the filter
would reject every candidate and the search would starve. It is ported because the
paper-faithful ReNeLLM row needs it — not because the IPI row should use it. Keep it
opt-in; see the recipe's "IPI adaptations vs original" section.
"""
from __future__ import annotations

import logging
import random
import re
from typing import Optional, Union

from ..datasets import AttackDataset, Instance
from ..seed import SeedTemplate
from .base import ConstraintBase

log = logging.getLogger(__name__)

__all__ = ["DeleteOffTopic", "DeleteHarmLess"]


def _constraint_prompt(method: str, variant: str = "original") -> str:
    return SeedTemplate().new_seeds(
        prompt_usage="constraint", method_list=[method], variant=variant)[0]


class DeleteOffTopic(ConstraintBase):
    r"""
    Drop candidates that no longer ask for the attacker's goal — TAP's Phase 1.

    A branching attacker drifts: three mutations in, a prompt can be an elaborate story
    that asks for nothing. Scoring those costs a victim query each, so TAP asks a cheap
    model "does this still ask for the same thing?" first and keeps the ``tree_width``
    best.

    Args:
        eval_model: Anything callable with a messages list — a ``UnifiedLLM`` is.
        tree_width: How many candidates survive.
        variant:    Registry variant of the judge prompt. ``original`` is upstream's
                    verbatim ``[[YES]]``/``[[NO]]`` prompt.
        seed:       RNG seed for the tie-breaking shuffle.

    Note:
        ``attacks/tap.py`` currently inlines its own Yes/No version of this
        (``constraint.TAP_on_topic``, authored here, different output format and prompt
        text). The recipe switches to this component in Phase H; until then the two
        coexist and are *not* interchangeable — the parsers differ.

    Deviations from upstream: the judge is prompted per instance with that instance's own
    ``query`` (upstream sets the system message once from ``dataset[0].query``, so a batch
    with mixed goals is judged entirely against the first one); and the all-off-topic
    fallback takes ``[:2]`` rather than ``[list[0], list[1]]``, which ``IndexError``s on a
    single-candidate dataset.
    """

    def __init__(self, eval_model, tree_width: int = 10, variant: str = "original",
                 seed: Optional[int] = None):
        self.eval_model = eval_model
        self.tree_width = tree_width
        self.system_prompt = _constraint_prompt("DeleteOffTopic", variant)
        self._rng = random.Random(seed)

    def get_evaluator_prompt_on_topic(self, attack_prompt: str) -> str:
        """The user turn handed to the judge. Verbatim from upstream."""
        return f"[PROMPT]:{attack_prompt}"

    def process_output_on_topic_score(self, raw_output: str) -> int:
        """
        1 if the judge said ``[[yes]]``, 0 if ``[[no]]``.

        An unparseable answer scores 1 — upstream's choice, and the safe one: a broken
        judge should not silently delete the whole tree.
        """
        match = re.search(r"\[\[(yes|no)\]\]", raw_output.lower())
        return int(match.group(1) == "yes") if match else 1

    def score(self, instance: Instance) -> int:
        """Ask the judge whether this candidate still targets ``instance.query``."""
        messages = [
            {"role": "system",
             "content": self.system_prompt.replace("{query}", instance.query or "")},
            {"role": "user",
             "content": self.get_evaluator_prompt_on_topic(instance.jailbreak_prompt or "")},
        ]
        try:
            raw_output = self.eval_model(messages)
        except Exception as exc:
            log.warning("[DeleteOffTopic] judge call failed, keeping candidate: %s", exc)
            return 1
        return self.process_output_on_topic_score(raw_output)

    def keep(self, instance: Instance, **kwargs) -> bool:
        return bool(self.score(instance))

    def __call__(self, dataset: AttackDataset, *args, **kwargs) -> AttackDataset:
        """Score every candidate, then keep the ``tree_width`` best on-topic ones."""
        scored = [(self.score(instance), instance) for instance in dataset]
        if not scored:
            return AttackDataset([])

        self._rng.shuffle(scored)                      # permute ties
        scored.sort(key=lambda pair: pair[0], reverse=True)

        width = min(self.tree_width, len(scored))
        truncated = [scored[i][1] for i in range(width) if scored[i][0] > 0]
        if not truncated:
            truncated = [instance for _, instance in scored[:2]]
        return AttackDataset(truncated)

    def __repr__(self) -> str:
        return f"DeleteOffTopic(tree_width={self.tree_width})"


class DeleteHarmLess(ConstraintBase):
    """
    Keep only candidates a model judges harmful — ReNeLLM's rewrite filter.

    See the module docstring for why our IPI ReNeLLM row does not use this.

    Args:
        eval_model:     Callable with a messages list, or with a plain string.
        prompt_pattern: Which instance text is judged. Default ``"{query}"``.
        attr_name:      Attributes the pattern references. Default ``["query"]``.
        variant:        Registry variant of the judge prompt.
    """

    def __init__(self, eval_model, prompt_pattern: Optional[str] = None,
                 attr_name: Optional[list[str]] = None, variant: str = "original"):
        self.eval_model = eval_model
        self._prompt = _constraint_prompt("DeleteHarmLess", variant)
        self._pattern = ["1"]
        self.prompt_pattern = "{query}" if prompt_pattern is None else prompt_pattern
        self.attr_name = list(attr_name or ["query"])

    def set_prompt(self, prompt: str):
        self._prompt = prompt

    def set_pattern(self, pattern: list[str]):
        self._pattern = list(pattern)

    def _format(self, instance: Instance) -> str:
        out = self.prompt_pattern
        for attr in self.attr_name:
            value = getattr(instance, attr, "") or ""
            if isinstance(value, (list, tuple)):
                value = value[-1] if value else ""
            out = out.replace("{" + attr + "}", str(value))
        return out

    def judge(self, seed: str) -> bool:
        """True when the judge's answer contains one of ``_pattern`` (default ``"1"``)."""
        if "{seed}" in self._prompt:
            text = self._prompt.format(seed=seed)
        else:
            text = self._prompt + seed
        try:
            outputs = self.eval_model(text)
        except Exception as exc:
            log.warning("[DeleteHarmLess] judge call failed, keeping candidate: %s", exc)
            return True
        return any(pattern in outputs for pattern in self._pattern)

    def keep(self, instance: Instance, **kwargs) -> bool:
        return self.judge(self._format(instance))
