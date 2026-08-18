"""
MutationBase — the ABC every mutation operator implements.

Mirrors ``easyjailbreak.mutation.mutation_base.MutationBase``: an operator takes an
``AttackDataset`` and returns a new one, recording parent/child lineage as it goes. That
lineage is not decoration — ``MCTSExploreSelectPolicy`` descends ``children``, and
``Instance.level`` is what makes a tree search possible at all.

    MutationBase.__call__(dataset) -> dataset      # 1-to-n, records parents/children

Two families, the same split upstream draws:

    generation.py   needs a model — the operator prompts an LLM to rewrite the text
    rule.py         deterministic — encodings, scrambles, sentence crossover, synonyms

Which field gets rewritten is ``attr_name`` (upstream's convention), default ``"query"``.
An operator that rewrites the *template* rather than the payload sets
``attr_name="jailbreak_prompt"``.

The scalar path
---------------
As in ``metrics/``, recipes have not migrated to the carrier yet (Phase H) and hold plain
strings. ``mutate(text)`` is the adapter: same transform, no ``Instance`` in sight. The
dataset path calls it too, so there is one implementation.

Deviations from upstream
------------------------
  * **The model is bound at construction** (upstream does this for its generation ops
    too), so ``mutate()`` takes only the text. Ours accepts any ``callable(str) -> str``,
    which a ``UnifiedLLM`` already is.
  * **The placeholder guard.** An operator that rewrites a template must not lose
    ``{query}`` — a template without it is un-attackable, and upstream relies purely on
    the LLM obeying "remember to have {query} in your answer". We keep that instruction
    *and* fall back to the input when the placeholder disappears.
  * **Empty output falls back to the input** rather than propagating "".
  * Upstream's ``Leetspeak`` appends the parent to ``new_instance.children`` instead of
    ``.parents`` — a typo that inverts the lineage edge. Ours goes through one
    ``new_child()`` helper, so every operator records it the same way.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from ..datasets import AttackDataset, Instance
from ..seed import PLACEHOLDER

__all__ = ["MutationBase", "LLMCallable"]

LLMCallable = Callable[[str], str]


class MutationBase(ABC):
    """
    A single text-transformation operator. See module docstring.

    Args:
        attr_name: Which ``Instance`` field to rewrite. Default ``"query"``.
    """

    #: Instance field this operator rewrites.
    attr_name: str = "query"

    #: Wrapper template attached to a mutated instance that has no ``jailbreak_prompt``
    #: yet (upstream behaviour). Lives in the seed registry under ``mutation.<Name>``.
    default_jailbreak_prompt: Optional[str] = None

    #: True when the operator rewrites a template that must keep ``{query}``. The guard
    #: only fires if the *input* had the placeholder, so one operator (Rephrase) can
    #: rewrite a GPTFuzzer template (guard active) or a ReNeLLM payload (inactive),
    #: exactly as upstream reuses it.
    preserves_placeholder: bool = False

    def __init__(self, attr_name: Optional[str] = None):
        if attr_name is not None:
            self.attr_name = attr_name

    # ------------------------------------------------------------------
    # Component seam
    # ------------------------------------------------------------------

    def __call__(self, dataset: AttackDataset, **kwargs) -> AttackDataset:
        """Mutate every instance; return the children as a new dataset."""
        mutated: list[Instance] = []
        for instance in dataset:
            mutated.extend(self._get_mutated_instance(instance, **kwargs))
        return AttackDataset(mutated)

    def _get_mutated_instance(self, instance: Instance, **kwargs) -> list[Instance]:
        """
        One parent -> one child, by default. Override for 1-to-n operators.
        """
        text = getattr(instance, self.attr_name, "") or ""
        child = self.new_child(instance)
        setattr(child, self.attr_name, self.mutate(text, instance=instance, **kwargs))
        if child.jailbreak_prompt is None and self.default_jailbreak_prompt is not None:
            child.jailbreak_prompt = self.default_jailbreak_prompt
        return [child]

    @staticmethod
    def new_child(instance: Instance) -> Instance:
        """
        Copy an instance and wire the lineage edge in both directions.

        Public because it is the *only* supported way to add a candidate to a search:
        it sets ``parents``/``children`` and bumps ``level``, which is what
        ``MCTSExploreSelectPolicy`` descends and discounts by. Recipes that branch
        without calling a mutation operator (TAP asks its attacker LLM for the next
        prompt) call this directly.
        """
        child = instance.copy()
        child.parents.append(instance)
        child.level = (instance.level or 0) + 1
        instance.children.append(child)
        return child

    # ------------------------------------------------------------------
    # The transform
    # ------------------------------------------------------------------

    def mutate(self, text: str, **kwargs) -> str:
        """
        Transform one string. The guards live here, so both paths get them.

        Falls back to the input when the operator returns nothing, or when it drops a
        ``{query}`` the input had.
        """
        out = self._mutate(text, **kwargs)
        out = (out or "").strip()
        if not out:
            return text
        if self.preserves_placeholder and PLACEHOLDER in text and PLACEHOLDER not in out:
            return text
        return out

    @abstractmethod
    def _mutate(self, text: str, **kwargs) -> str:
        """The operator's actual transform, unguarded."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{type(self).__name__}(attr_name={self.attr_name!r})"


class LLMMutation(MutationBase):
    """
    Base for the ``generation`` family: an operator that prompts a model to rewrite text.

    Args:
        model: Anything callable as ``model(str) -> str`` — a ``UnifiedLLM`` is one, and
               so is ``lambda t: attacker(t)``. Bound at construction, as upstream does.
        attr_name: Which ``Instance`` field to rewrite.
    """

    def __init__(self, model: Union[LLMCallable, object], attr_name: Optional[str] = None):
        super().__init__(attr_name)
        if not callable(model):
            raise TypeError(
                f"{type(self).__name__} needs a callable(str) -> str (a UnifiedLLM is "
                f"one), got {type(model).__name__}")
        self.model: LLMCallable = model

    def _mutate(self, text: str, **kwargs) -> str:
        return self.model(self._prompt(text, **kwargs))

    @abstractmethod
    def _prompt(self, text: str, **kwargs) -> str:
        """The instruction handed to the model. Verbatim from upstream."""

    def __repr__(self) -> str:
        model = getattr(self.model, "model_name", None) or type(self.model).__name__
        return f"{type(self).__name__}(model={model})"
