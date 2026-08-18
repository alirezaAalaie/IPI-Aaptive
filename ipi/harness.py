"""
The plumbing between an ``Instance`` and a ``Victim`` — what an attack needs to *run*.

The symmetric half of ``ipi.metrics``, which owns what an attack needs to be *judged*.
Nothing here decides success.

    make_target_fn(instance, victim)      the ``target_fn(injection) -> response``
                                          callable every recipe is written against
    build_victim_messages(...)            the messages ``target_fn`` would send
    build_optimization_messages(...)      the same, plus the defense's own preprocessing
    split_optimization_prompt(...)        that prompt, split around the adversarial span,
                                          for the token-level attacks
    attack_context(instance)              the ``context`` dict TAP / PAIR / run_attack take
    resolve_optimization_target(instance) the literal token sequence GCG / RS / Beam-RS
                                          optimize toward

The last three exist because GCG, BEAST and AutoDAN each used to build their own
prompt — a bare ``[system][user]`` pair — while judging success through
``make_target_fn``'s IPI carrier. They were optimizing a prompt the victim never saw.
Anything white-box must go through this seam.

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

__all__ = [
    "make_target_fn", "attack_context", "resolve_optimization_target",
    "build_victim_messages", "build_optimization_messages",
    "split_optimization_prompt",
]


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
    data_separator: str = "\n\n",
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
        data_separator:         What sits between the legitimate content and the
                                injection inside the data channel. ``"\n\n"`` is the
                                house default. The OPI static attacks override it,
                                because for them the separator *is* the attack — the
                                paper distinguishes ``naive`` from ``escape`` by
                                exactly one space versus one newline, and a fixed
                                ``"\n\n"`` collapses that distinction.

    Returns:
        Callable[[str], str] — target_fn(injection_string) -> response_string.
    """
    attrs        = instance.attack_attrs
    user_task    = attrs.get("user_task", "")
    tool_schema  = attrs.get("tool_schema", "")
    context      = attrs.get("pipeline_context", "")

    def target_fn(injection: str) -> str:
        messages = build_victim_messages(
            instance, victim, injection, system_prompt_template, data_separator)
        try:
            return victim.generate(messages)
        except Exception as e:
            log.warning("[target_fn] instance=%s error: %s", instance.id, e)
            return ""

    return target_fn


def build_victim_messages(
    instance: Instance,
    victim: Victim,
    injection: str,
    system_prompt_template: str = "",
    data_separator: str = "\n\n",
) -> list[dict]:
    """
    The messages list ``make_target_fn`` hands to ``victim.generate``.

    Split out of ``make_target_fn`` so the white-box attacks can see the same prompt
    shape without calling the victim. This is the *undefended* list: a
    ``DefendedVictim`` applies its own ``preprocess_messages`` inside ``generate``,
    so applying it here too would double it. Use ``build_optimization_messages`` when
    you want the post-defense shape.
    """
    attrs       = instance.attack_attrs
    user_task   = attrs.get("user_task", "")
    tool_schema = attrs.get("tool_schema", "")
    context     = attrs.get("pipeline_context", "")

    tmpl = system_prompt_template or getattr(victim, "system_prompt_template", "")
    if tmpl:
        system_prompt = tmpl.format(user_task=user_task, tool_schema=tool_schema)
    else:
        system_prompt = victim.system_prompt

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if user_task:
        # User Task + Data in a single user message (Instruction -> User Task -> Data).
        # Two consecutive 'user' turns are rejected by most chat APIs.
        data_str = (
            f"{context}{data_separator}{injection}" if context
            else f"<env>\n{injection}\n</env>"
        )
        user_content = f"User Task:\n{user_task}\n\nContext:\n{data_str}"
        messages.append({"role": "user", "content": user_content})
    else:
        # No user task (hand-built instance) — inject directly.
        messages.append({"role": "user", "content": injection})

    return messages


def build_optimization_messages(
    instance: Instance,
    victim: Victim,
    injection: str,
    system_prompt_template: str = "",
) -> list[dict]:
    """
    The messages the victim's *model* actually sees — carrier plus any defense.

    ``build_victim_messages`` then the victim's own ``preprocess_messages`` hook, which
    is where a ``DefendedVictim`` rewrites the channels. GCG / BEAST / AutoDAN optimize
    against this, so a white-box row measures the attack against the defense's real
    prompt rather than the undefended one.

    A defense whose preprocessing is input-dependent makes this a snapshot, not an
    invariant: the attacks build it once per scenario.
    """
    messages = build_victim_messages(instance, victim, injection, system_prompt_template)
    pre = getattr(victim, "preprocess_messages", None)
    if callable(pre):
        try:
            return pre(messages)
        except Exception as e:
            log.warning(
                "[build_optimization_messages] instance=%s preprocess_messages raised "
                "(%s) — optimizing against the undefended prompt shape", instance.id, e)
    return messages


def split_optimization_prompt(
    instance: Instance,
    victim: Victim,
    injection_prefix: str,
    injection_suffix: str = "",
    tokenizer=None,
    system_prompt_template: str = "",
) -> tuple[str, str, bool]:
    """
    Decompose the victim's whole prompt around where adversarial tokens will go.

    Returns ``(head_text, tail_text, add_special_tokens)`` such that

        encode(head) + <adversarial ids> + encode(tail) [+ encode(target)]

    is exactly what the victim's model would be fed for
    ``injection_prefix + <adv> + injection_suffix``. ``tail`` therefore carries the
    close of the user turn and the generation prompt — BEAST's ``end_inst_token``.

    This is the fix for the two failure modes the white-box attacks shared: optimizing
    a bare ``[system][user]`` prompt while success was judged through the IPI carrier,
    and appending adversarial tokens *after* the generation prompt (putting them in
    the assistant turn). Both are silent.

    Falls back to the undefended shape, with a warning, if a defense's preprocessing
    eats the marker.
    """
    from .llm_unified import ADV_SENTINEL, split_prompt_around

    tokenizer = tokenizer if tokenizer is not None else victim.tokenizer
    marked = f"{injection_prefix}{ADV_SENTINEL}{injection_suffix}"

    messages = build_optimization_messages(
        instance, victim, marked, system_prompt_template)
    try:
        return split_prompt_around(tokenizer, messages, ADV_SENTINEL)
    except ValueError as e:
        log.warning(
            "[split_optimization_prompt] instance=%s: %s — the defense's "
            "preprocess_messages does not preserve the injected span verbatim; "
            "falling back to the undefended prompt shape", instance.id, e)
    messages = build_victim_messages(
        instance, victim, marked, system_prompt_template)
    return split_prompt_around(tokenizer, messages, ADV_SENTINEL)
