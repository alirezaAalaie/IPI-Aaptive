"""
Instance — the carrier object every attack component speaks.

Ported from EasyJailbreak's ``easyjailbreak/datasets/instance.py``. This is the seam
that makes mutation / constraint / selector / metrics interchangeable: each family
takes and returns ``AttackDataset``, and reads or writes fields on the ``Instance``
objects inside it.

    MutationBase.__call__(dataset)  -> dataset   # 1-to-n, records parents/children
    ConstraintBase.__call__(dataset)-> dataset   # filters
    SelectPolicy.select()           -> dataset   # + .update(dataset)
    Evaluator.__call__(dataset)                  # writes instance.eval_results

Field semantics (IPI reading of the upstream names)
---------------------------------------------------
    query               The attacker's goal — the string that fills a seed template's
                        ``{query}`` placeholder. Upstream this is the harmful request;
                        for IPI it is ``injection_goal`` (e.g. "Print only HACKED").
    jailbreak_prompt    The injection template/string. Set by the attack.
    reference_responses What a successful attack should produce (``target_str``).
                        Consumed by the Match / PrefixExactMatch evaluators.
    target_responses    What the victim actually returned.
    eval_results        Per-response verdicts written by an Evaluator.
    parents / children  Lineage, populated by mutations. MCTS descends ``children``.
    index / visited_num / level
                        Search bookkeeping. Required by MCTSExploreSelectPolicy and
                        UCBSelectPolicy — the absence of these fields is precisely why
                        the old GPTFuzzer could only do flat UCB1.
    attack_attrs        Free-form dict. Carries the IPI-specific fields that have no
                        upstream counterpart: user_task, user_target, user_eval_mode,
                        attack_eval_mode, optimization_target, clean_context,
                        pipeline_context, tool_schema.

Deviations from upstream
------------------------
  * ``id`` is a first-class field. Our results are reported per scenario id, and the
    dual-verifiable dataset has stable ids; upstream has no equivalent.
  * ``attack_attrs`` defaults carry the IPI keys rather than upstream's
    ``{'Mutation': None, 'query_class': None}``.

Attribute access is backed by a ``_data`` dict (upstream behaviour), so components can
attach arbitrary attributes — ``instance.visited_num = 0`` — without a schema change.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

__all__ = ["Instance"]


class Instance:
    """One attack candidate / scenario. See module docstring for field semantics."""

    def __init__(
        self,
        *,
        id: str = "",
        query: Optional[str] = None,
        jailbreak_prompt: Optional[str] = None,
        reference_responses: Optional[list[str]] = None,
        target_responses: Optional[list[str]] = None,
        eval_results: Optional[list] = None,
        parents: Optional[list] = None,
        children: Optional[list] = None,
        attack_attrs: Optional[dict] = None,
        **kwargs,
    ):
        # `_data` must be set first — __setattr__ routes everything else into it.
        self._data: dict[str, Any] = {}

        self.id = id
        self.query = query
        self.jailbreak_prompt = jailbreak_prompt
        self.reference_responses = [] if reference_responses is None else reference_responses
        self.target_responses = [] if target_responses is None else target_responses
        self.eval_results = [] if eval_results is None else eval_results
        self.parents = [] if parents is None else parents
        self.children = [] if children is None else children

        # Search bookkeeping — selectors set these; defaults keep them always present.
        self.index: Optional[int] = None
        self.visited_num: int = 0
        self.level: int = 0

        self.attack_attrs = {} if attack_attrs is None else attack_attrs

        self._data.update(**kwargs)

    # ------------------------------------------------------------------
    # Derived counts (upstream API — selectors call these for reward)
    # ------------------------------------------------------------------

    @property
    def num_query(self) -> int:
        """Number of victim responses collected for this instance."""
        return len(self.target_responses)

    @property
    def num_jailbreak(self) -> int:
        """
        Number of responses an Evaluator marked successful.

        ``sum(eval_results)`` verbatim from upstream — selectors use this as the reward
        numerator, so it must stay a sum, not a truthy count. That means it is only
        meaningful when ``eval_results`` holds **binary** verdicts (a ``*Judge`` /
        ``*Match`` evaluator). Pairing a 1-10 ``*GetScore`` evaluator with a
        reward-driven selector would sum scores instead; use ``SelectBasedOnScores``
        for those, as upstream's TAP does.
        """
        return sum(self.eval_results)

    @property
    def num_reject(self) -> int:
        """Number of responses an Evaluator marked unsuccessful."""
        return len(self.eval_results) - self.num_jailbreak

    # ------------------------------------------------------------------
    # Copying / serialisation
    # ------------------------------------------------------------------

    def copy(self) -> "Instance":
        """
        Shallow-copy the instance.

        Lineage lists hold *references* (a copy shares its parents/children with the
        original, as upstream does) — deep-copying them would clone the whole search
        tree on every mutation.
        """
        new = Instance(
            id=self.id,
            query=self.query,
            jailbreak_prompt=self.jailbreak_prompt,
            reference_responses=list(self.reference_responses),
            target_responses=list(self.target_responses),
            eval_results=list(self.eval_results),
            parents=list(self.parents),
            children=list(self.children),
            attack_attrs=copy.deepcopy(self.attack_attrs),
        )
        rest = {k: v for k, v in self._data.items() if k not in new._data}
        new._data.update(copy.deepcopy(rest))
        return new

    def to_dict(self) -> dict:
        """Dict view excluding lineage (which is not serialisable — it is a graph)."""
        return {k: v for k, v in self._data.items() if k not in ("parents", "children")}

    def delete(self, *keys):
        """Remove attributes from the instance."""
        for key in keys:
            del self._data[key]

    # ------------------------------------------------------------------
    # Attribute plumbing (upstream behaviour: arbitrary attrs land in _data)
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        # Only called when normal lookup fails, so `_data` itself is safe here
        # provided __init__ set it first.
        try:
            return self.__dict__["_data"][name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:
        q = (self.query or "")[:40]
        return (
            f"Instance(id={self.id!r}, query={q!r}, "
            f"n_responses={len(self.target_responses)}, level={self.level})"
        )
