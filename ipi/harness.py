"""
The plumbing between an ``Instance`` and a ``Victim`` — what an attack needs to *run*.

The symmetric half of ``ipi.metrics``, which owns what an attack needs to be *judged*.
Nothing here decides success.

    VictimQuery(instance, victim)         the victim as a *component* —
                                          ``dataset -> dataset``, the same shape as
                                          mutation / constraint / evaluator, so a
                                          recipe's loop reads as one pipeline
    make_target_fn(instance, victim)      the ``target_fn(injection) -> response``
                                          callable the unconverted recipes are written
                                          against; ``VictimQuery`` wraps it
    build_channeled_prompt(...)           the instruction/data split for one injection
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
from dataclasses import dataclass
from typing import Callable, Optional

from .channels import ChanneledPrompt
from .datasets import AttackDataset, Instance
from .victim import Victim

log = logging.getLogger(__name__)

__all__ = [
    "VictimQuery",
    "make_target_fn", "attack_context", "resolve_optimization_target",
    "build_channeled_prompt", "build_victim_messages",
    "build_optimization_messages", "split_optimization_prompt",
    "apply_defense_pipeline", "rebuild_victim_prompt", "TargetFnSpec",
]


@dataclass(frozen=True)
class TargetFnSpec:
    """
    How a ``target_fn`` builds its prompt — enough to rebuild any one of its calls.

    ``make_target_fn`` records this on the victim so the reporting layer can
    reconstruct the prompt for the injection an attack finally *reported*, rather
    than reading back whichever call happened to be last. The two differ for every
    search attack: TAP's last query is its last candidate, not its best one.

    Deliberately the build parameters and not the prompts themselves — a search
    attack makes thousands of calls, and retaining a rendered prompt per call is
    tens of MB per scenario for an audit trail that only ever needs one of them.
    """
    instance_id: str
    system_prompt_template: str
    data_separator: str


def apply_defense_pipeline(victim: Victim, messages: list[dict]) -> list[dict]:
    """
    Run ``messages`` through every ``preprocess_messages`` in a defense chain.

    Mirrors what ``DefendedVictim.generate`` does by nesting: each wrapper
    preprocesses, then hands the result to its target, which preprocesses again.
    Calling ``victim.preprocess_messages`` once — what ``build_optimization_messages``
    used to do — only applies the **outermost** hook, and ``CompositeDefense``'s own
    hook is a no-op (its work happens in the defenses it chains underneath). So a
    white-box attack against ``CompositeDefense(t, [Instructional, Sandwich, Reminder])``
    was handed a prompt with none of the three applied — the undefended shape, under a
    defended label, silently.

    Stops at the first node with no ``target`` (``TargetLLM`` holds its model in
    ``llm`` and has no hook), and guards against a cycle.
    """
    seen: set[int] = set()
    node = victim
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        pre = getattr(node, "preprocess_messages", None)
        if callable(pre):
            messages = pre(messages)
        node = getattr(node, "target", None)
    return messages


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
    # Recorded so the reporting layer can rebuild the prompt for the injection the
    # attack finally reports, instead of reading back whichever call was last.
    try:
        victim.last_target_fn_spec = TargetFnSpec(
            instance_id=instance.id,
            system_prompt_template=system_prompt_template,
            data_separator=data_separator,
        )
    except Exception:  # pragma: no cover — a victim with __slots__ and no such slot
        log.debug("[make_target_fn] could not record the build spec on %s",
                  type(victim).__name__)

    def target_fn(injection: str) -> str:
        messages = build_victim_messages(
            instance, victim, injection, system_prompt_template, data_separator)
        try:
            return victim.generate(messages)
        except Exception as e:
            log.warning("[target_fn] instance=%s error: %s", instance.id, e)
            return ""

    return target_fn


class VictimQuery:
    """
    The victim as a component: ``AttackDataset -> AttackDataset``.

    The fifth member of the family ``mutation`` / ``constraint`` / ``selector`` /
    ``metrics`` already form, and the reason a recipe's main loop can now be read as
    one pipeline instead of a loop with plumbing inlined in it::

        dataset = self.mutator(dataset)      # branch
        dataset = self.constraint(dataset)   # prune before spending queries
        dataset = self.query(dataset)        # ← this: ask the victim
        self.evaluator(dataset)              # score
        dataset = self.selector.select(dataset)

    Before this existed the third line was six lines of ``try``/``except`` plus a
    ``n_queries += 1``, hand-written in eleven recipes, each with its own idea of what
    to do when the victim raised.

    What it owns
    ------------
    * **The prompt.** It wraps ``make_target_fn``, so every query goes through the one
      place the IPI carrier is defined. A recipe never assembles messages.
    * **The query count.** ``n_queries`` is calls to the victim, the definition the
      whole results table uses. Model forward/backward passes are *not* counted here;
      they belong in ``n_forward_passes``.
    * **The budget.** ``budget`` caps queries. ``__call__`` returns only the candidates
      it actually reached, so a half-spent budget truncates the batch rather than
      handing the evaluator instances with no response on them.
    * **Failure.** ``make_target_fn``'s callable already turns an exception into ``""``
      and logs it; nothing downstream sees a partially-filled dataset.

    Args:
        instance:  The *scenario* being attacked — the source of the IPI context
                   (user_task, pipeline_context, tool_schema). Not to be confused with
                   the candidate instances flowing through ``__call__``, which carry
                   the injections.
        victim:    The Victim (TargetLLM or any defense wrapping one).
        attr_name: Which candidate field holds the injection. ``"jailbreak_prompt"``,
                   matching the mutation family's default for template-rewriting ops.
        render:    Optional ``Callable[[Instance], str]`` producing the string actually
                   sent, when the candidate field is a *template* rather than a finished
                   injection (GPTFuzzer's seeds carry ``{query}``). The rendered string
                   is written back to ``attack_attrs["injection"]`` so the reported
                   ``attack_str`` is what the victim saw, not the template behind it.
        budget:    Max victim queries, or ``None`` for unlimited.
        system_prompt_template / data_separator:
                   Passed straight to ``make_target_fn``. The separator is the attack
                   for the OPI one-shots (one space versus one newline), so it is a
                   per-recipe choice, not a house constant.
    """

    def __init__(
        self,
        instance: Instance,
        victim: Victim,
        *,
        attr_name: str = "jailbreak_prompt",
        render: Optional[Callable[[Instance], str]] = None,
        budget: Optional[int] = None,
        system_prompt_template: str = "",
        data_separator: str = "\n\n",
    ):
        self.instance = instance
        self.victim = victim
        self.attr_name = attr_name
        self.render = render
        self.budget = budget
        self.n_queries = 0
        self.target_fn = make_target_fn(
            instance, victim, system_prompt_template, data_separator)

    # ------------------------------------------------------------------
    # The seam
    # ------------------------------------------------------------------

    def __call__(self, dataset: AttackDataset, **kwargs) -> AttackDataset:
        """
        Query the victim once per candidate, writing the reply to
        ``target_responses`` and the string actually sent to
        ``attack_attrs["injection"]``.

        Returns the candidates that were reached — the whole dataset unless the budget
        ran out mid-batch. Returning the short list rather than the full one is
        deliberate: an instance with an empty ``target_responses`` scored by an
        evaluator reads as a failed attack rather than as an unasked question.
        """
        reached = []
        for candidate in dataset:
            if self.exhausted:
                break
            injection = self.injection_of(candidate)
            candidate.attack_attrs["injection"] = injection
            candidate.target_responses = [self.ask(injection)]
            reached.append(candidate)
        return AttackDataset(reached)

    def ask(self, injection: str) -> str:
        """One query, counted. ``""`` if the victim raised (``target_fn`` logs it)."""
        self.n_queries += 1
        return self.target_fn(injection)

    def injection_of(self, candidate: Instance) -> str:
        """The string to send for ``candidate`` — ``render`` if given, else the field."""
        if self.render is not None:
            return self.render(candidate)
        return getattr(candidate, self.attr_name, "") or ""

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    @property
    def exhausted(self) -> bool:
        """True once the query budget is spent. Always False when there is no budget."""
        return self.budget is not None and self.n_queries >= self.budget

    @property
    def remaining(self) -> Optional[int]:
        """Queries left, or ``None`` when unbudgeted."""
        if self.budget is None:
            return None
        return max(0, self.budget - self.n_queries)

    def __repr__(self) -> str:
        budget = "unbudgeted" if self.budget is None else f"{self.n_queries}/{self.budget}"
        return (f"VictimQuery(victim={type(self.victim).__name__}, "
                f"instance={self.instance.id!r}, queries={budget})")


def rebuild_victim_prompt(
    instance: Instance,
    victim: Victim,
    injection: str,
) -> Optional[list[dict]]:
    """
    Rebuild the exact prompt the model saw for ``injection`` — carrier plus the
    whole defense chain.

    Uses the ``TargetFnSpec`` ``make_target_fn`` left on the victim, so the
    separator an attack chose is honoured (the OPI static attacks differ from each
    other by exactly that: one space versus one newline). Falls back to the house
    defaults when the spec is absent or belongs to a different instance.

    Returns ``None`` if the prompt cannot be rebuilt — a caller reporting an audit
    trail should then fall back to what the victim recorded, and say which it used.
    Rebuilding runs the defenses' preprocessing again but never queries the model.
    """
    spec = getattr(victim, "last_target_fn_spec", None)
    tmpl, sep = "", "\n\n"
    if isinstance(spec, TargetFnSpec) and spec.instance_id == instance.id:
        tmpl, sep = spec.system_prompt_template, spec.data_separator
    try:
        return build_optimization_messages(instance, victim, injection, tmpl, sep)
    except Exception as e:
        log.warning(
            "[rebuild_victim_prompt] instance=%s could not rebuild the reported "
            "prompt (%s)", instance.id, e)
        return None


def build_channeled_prompt(
    instance: Instance,
    victim: Victim,
    injection: str,
    system_prompt_template: str = "",
    data_separator: str = "\n\n",
) -> ChanneledPrompt:
    """
    The prompt for one injection, with the trust boundary kept as structure.

    This is where the IPI prompt shape is *defined*; ``build_victim_messages`` only
    renders it. The two channels are read straight off the instance —
    ``user_task`` is trusted, ``pipeline_context`` plus the injection are not — so
    no defense downstream has to work out which is which. Before this existed,
    six defenses re-derived the split from the rendered text with a regex, and one
    prompt-shape change silently put the user's own task in the untrusted channel.

    ``data_separator`` is what sits between the legitimate content and the
    injection *inside* the data channel; the OPI static attacks override it,
    because for them the separator is the attack (``naive`` differs from
    ``escape`` by one space versus one newline).
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

    if not user_task:
        # No user task (hand-built instance) — the injection is the whole user turn,
        # and all of it is untrusted.
        return ChanneledPrompt(system=system_prompt, instruction="", data=injection)

    # User Task + Data in a single user message (Instruction -> User Task -> Data).
    # Two consecutive 'user' turns are rejected by most chat APIs, which is why the
    # split has to travel beside the text rather than in it.
    if context:
        return ChanneledPrompt(
            system=system_prompt,
            instruction=user_task,
            data=f"{context}{data_separator}{injection}",
        )
    return ChanneledPrompt(
        system=system_prompt,
        instruction=user_task,
        data=injection,
        data_prefix="<env>\n",
        data_suffix="\n</env>",
    )


def build_victim_messages(
    instance: Instance,
    victim: Victim,
    injection: str,
    system_prompt_template: str = "",
    data_separator: str = "\n\n",
) -> list[dict]:
    """
    The messages list ``make_target_fn`` hands to ``victim.generate``.

    A ``ChanneledMessages`` — an ordinary ``list[dict]`` that additionally carries
    the ``ChanneledPrompt`` it was rendered from, so a defense reads the
    instruction/data split instead of parsing for it.

    Split out of ``make_target_fn`` so the white-box attacks can see the same prompt
    shape without calling the victim. This is the *undefended* list: a
    ``DefendedVictim`` applies its own ``preprocess_messages`` inside ``generate``,
    so applying it here too would double it. Use ``build_optimization_messages`` when
    you want the post-defense shape.
    """
    return build_channeled_prompt(
        instance, victim, injection, system_prompt_template, data_separator,
    ).to_messages()


def build_optimization_messages(
    instance: Instance,
    victim: Victim,
    injection: str,
    system_prompt_template: str = "",
    data_separator: str = "\n\n",
) -> list[dict]:
    """
    The messages the victim's *model* actually sees — carrier plus **every** defense.

    ``build_victim_messages`` then ``apply_defense_pipeline``. GCG / BEAST / AutoDAN
    optimize against this, so a white-box row measures the attack against the defense's
    real prompt rather than the undefended one.

    It used to call ``victim.preprocess_messages`` once, which applies only the
    outermost hook — and ``CompositeDefense``'s own hook is a no-op, so a chain of three
    in-context defenses contributed *nothing* and the attack optimized the undefended
    prompt under a defended label. ``apply_defense_pipeline`` walks the chain the way
    ``generate`` nests it.

    A defense whose preprocessing is input-dependent makes this a snapshot, not an
    invariant: the attacks build it once per scenario.
    """
    messages = build_victim_messages(
        instance, victim, injection, system_prompt_template, data_separator)
    try:
        return apply_defense_pipeline(victim, messages)
    except Exception as e:
        log.warning(
            "[build_optimization_messages] instance=%s a defense's preprocess_messages "
            "raised (%s) — optimizing against the undefended prompt shape",
            instance.id, e)
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
