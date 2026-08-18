"""
The plumbing between an ``Instance`` and a ``Victim`` — what an attack needs to *run*.

The symmetric half of ``ipi.metrics``, which owns what an attack needs to be *judged*.
Nothing here decides success.

    make_target_fn(instance, victim)      the ``target_fn(injection) -> response``
                                          callable every recipe is written against
    attack_context(instance)              the ``context`` dict TAP / PAIR / run_attack take
    resolve_optimization_target(instance) the literal token sequence GCG / RS / Beam-RS
                                          optimize toward

This is the heir of the deleted ``ipi/evaluator.py``. It differs in one way that matters:
the seam is now ``Instance``, not the deleted ``IPIScenario``. The IPI-specific fields an
attack needs — ``user_task``, ``pipeline_context``, ``tool_schema`` — have no upstream
counterpart on ``Instance``, so they live in ``attack_attrs`` and are read through the
accessors below rather than by attribute. Reading them by hand is how a typo becomes a
silently empty prompt: ``Instance.__getattr__`` raises, but ``attack_attrs.get`` does not.

``ipi/evaluator.py``'s other two functions were RS-specific and moved into
``attacks/adaptive.py``, their only caller.
"""
from __future__ import annotations

import logging
from typing import Callable

from .datasets import Instance
from .victim import Victim

log = logging.getLogger(__name__)

__all__ = ["make_target_fn", "attack_context", "resolve_optimization_target"]


def attack_context(instance: Instance) -> dict:
    """
    The ``context`` dict ``run_tap`` / ``run_pair`` / ``run_attack`` expect.

    Replaces the deleted ``IPIScenario.to_attack_context()``. The key names are the
    attacker-side ones those functions already document, so they are deliberately *not*
    renamed to match ``attack_attrs``.
    """
    attrs = instance.attack_attrs
    return {
        "user_task":            attrs.get("user_task", ""),
        "tool_schema":          attrs.get("tool_schema", ""),
        "target_tool_calls":    attrs.get("target_str", ""),
        "conversation_history": attrs.get("pipeline_context", ""),
    }


def resolve_optimization_target(instance: Instance) -> str:
    """
    The string a gradient / logprob attack optimizes toward.

    ``optimization_target`` first, because it is guaranteed to be a real token sequence.
    ``target_str`` is the fallback and is only safe when it is a literal — it may be a
    sentinel (``"__base64__"``) that evaluators resolve with a rule instead, and no
    tokenizer can optimize toward that. ``query`` is the last resort.

    Judge-guided attacks (TAP, PAIR) ignore this entirely and use their judge's score.
    """
    attrs = instance.attack_attrs
    return (
        attrs.get("optimization_target")
        or attrs.get("target_str")
        or (instance.query or "")
    )


def make_target_fn(
    instance: Instance,
    victim: Victim,
    system_prompt_template: str = "",
) -> Callable[[str], str]:
    """
    Build the ``target_fn`` for one instance.

    The returned callable takes an injection string, embeds it in the untrusted data
    position of the victim's prompt, and returns the victim's response. This is the
    single place the IPI prompt shape is defined, which is why every recipe goes through
    it rather than assembling messages itself.

    Args:
        instance:               The Instance being attacked.
        victim:                 Victim (TargetLLM or any defense subclass).
        system_prompt_template: Optional f-string template for the system prompt.
                                Available variables: {user_task}, {tool_schema}.
                                Empty means "use ``victim.system_prompt`` as-is"; the
                                victim's own ``system_prompt_template`` is consulted
                                first, so ``TargetLLM`` can fill in {tool_schema} per
                                instance without the caller passing anything.

    Returns:
        Callable[[str], str] — target_fn(injection_string) -> response_string.
    """
    attrs        = instance.attack_attrs
    user_task    = attrs.get("user_task", "")
    tool_schema  = attrs.get("tool_schema", "")
    context      = attrs.get("pipeline_context", "")

    tmpl = system_prompt_template or getattr(victim, "system_prompt_template", "")
    if tmpl:
        system_prompt = tmpl.format(user_task=user_task, tool_schema=tool_schema)
    else:
        system_prompt = victim.system_prompt

    def target_fn(injection: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if user_task:
            # User Task + Data in a single user message (Instruction -> User Task -> Data).
            # Two consecutive 'user' turns are rejected by most chat APIs.
            data_str = f"{context}\n\n{injection}" if context else f"<env>\n{injection}\n</env>"
            user_content = f"User Task:\n{user_task}\n\nContext:\n{data_str}"
            messages.append({"role": "user", "content": user_content})
        else:
            # No user task (hand-built instance) — inject directly.
            messages.append({"role": "user", "content": injection})

        try:
            return victim.generate(messages)
        except Exception as e:
            log.warning("[target_fn] instance=%s error: %s", instance.id, e)
            return ""

    return target_fn
