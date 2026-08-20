"""
ICA (In-Context Attack) adapted for Indirect Prompt Injection.

Original paper: Wei et al. (2023), "Jailbreak and Guard Aligned Language Models with
Only Few In-Context Demonstrations"
arXiv: https://arxiv.org/abs/2310.06387

Mechanism (faithful to the original)
------------------------------------
ICA prepends a handful of *demonstrations* in which the assistant complies with a
harmful request, then issues the real request. The reference implementation builds
these demonstrations from a fixed 5-shot template (AdvBench-style harmful goals paired
with affirmative "Sure, here is ..." completions), then appends ``User:{query}``.

We reproduce that construction directly:
  - The demonstration pool comes from the seed registry (``seed.ica_demos``).
  - The transcript is ``User:{goal}\nAssistant:{target}`` repeated ``prompt_num`` times,
    then the attacker goal.

Composition
-----------
One transcript, one query, no search — the whole algorithm is three lines inside
``single_attack``::

    query   = VictimQuery(instance, target)         # owns the call and the count
    dataset = query(AttackDataset([candidate]))
    EvaluatorIPISuccess(mode=eval_mode)(dataset)    # ground truth, not the judge

IPI adaptation vs original
--------------------------
  Delivery: the original ICA sends the demonstrations as **real alternating chat turns**
  to a raw chat model — a *direct* prompt-injection setting. For **indirect** prompt
  injection the payload must travel through the data channel, so the whole transcript
  (demos + goal) is emitted as one injected-document string and delivered by
  ``harness.VictimQuery`` as untrusted content.

  Demonstrations: ``variant="ipi"`` (default) uses ``demo.ICA.ipi`` — 10 authored pairs in
  which the assistant *actually carries out* an injected instruction (print a canary,
  echo a URL, adopt an identity prefix) or breaks one of its own rules. The paper's 30
  AdvBench pairs (``variant="original"``) only restate the request ("Sure, here is a
  tutorial on …") without doing anything, and their subject matter is mismatched to IPI,
  where the barrier is instruction *provenance* rather than refusal. ``variant="original"``
  stays reachable for the paper-reproduction row.

  success: resolved from the scenario (``resolve_attack_target``) so it matches the
           dataset's own ``attack_eval_mode`` (e.g. DualVerifiableDataset uses
           ``startswith`` / ``contains`` against a literal target string).

  score:   with no ``judge=``, the reported 1-10 score is the binary verdict mapped
           through ``Evaluator.as_score`` (10 / 1) rather than the hard 0 the deleted
           ``run_ica`` returned. The judge only annotates; ``AttackEvaluator``
           recomputes success.

Report of differences from the reference is in docs/easyjailbreak-audit.md.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..attacker import StaticAttacker
from ..datasets import AttackDataset, Instance
from ..harness import VictimQuery
from ..metrics import Evaluator, EvaluatorIPISuccess, resolve_attack_target
from ..victim import Victim
from ..seed import ica_demos

log = logging.getLogger(__name__)

DEFAULT_PROMPT_NUM = 10   # the whole demo.ICA.ipi pool (paper's own default is 5-shot)
DEFAULT_VARIANT = "ipi"


def build_ica_injection(
    goal: str,
    prompt_num: int = DEFAULT_PROMPT_NUM,
    demos: Optional[list[tuple[str, str]]] = None,
    variant: str = DEFAULT_VARIANT,
    seed: Optional[int] = None,
) -> str:
    """
    Build the ICA injection: ``prompt_num`` compliance demonstrations followed by the
    attacker goal, formatted exactly like the reference template
    (``User:{q}\\nAssistant:{a}`` ... ``User:{goal}``).

    Args:
        goal:       The attacker instruction placed after the demonstrations.
        prompt_num: Number of in-context demonstrations. Default 10 (the whole IPI pool).
        demos:      Optional (goal, target) pairs to use instead of the registry pool.
        variant:    Demonstration pool — "ipi" (default) or "original" (30 AdvBench pairs).
        seed:       RNG seed for sampling the demonstrations (reproducibility).
    """
    if demos is None:
        demos = ica_demos(n=prompt_num, variant=variant, seed=seed)
    demos = demos[:prompt_num]

    lines: list[str] = []
    for demo_goal, demo_target in demos:
        lines.append(f"User:{demo_goal}")
        lines.append(f"Assistant:{demo_target}")
    lines.append(f"User:{goal}")
    return "\n".join(lines)


class ICAAttacker(StaticAttacker):
    """
    In-Context Attack (ICA) — single-query, demonstration-based, API-compatible.

    From: Wei et al. (2023), "Jailbreak and Guard Aligned Language Models with Only Few
    In-Context Demonstrations".

    Args:
        judge:      Optional guidance Evaluator (ipi.metrics). Annotates the reported
                    candidate with a 1-10 score; it never decides success.
        prompt_num: Number of compliance demonstrations. Default 10 (the whole IPI pool);
                    the paper's own default is 5-shot over AdvBench.
        variant:    Demonstration pool — "ipi" (default) or "original" (30 AdvBench pairs,
                    for the paper-reproduction row). See the module docstring.
        eval_mode:  check_ipi_success mode. Default None → auto-detect from the scenario's
                    ``attack_eval_mode`` (e.g. DualVerifiableDataset).
        seed:       RNG seed for reproducible demo sampling.
    """

    _ATTACK_NAME = "ica"

    def __init__(
        self,
        judge: Optional[Evaluator] = None,
        prompt_num: int = DEFAULT_PROMPT_NUM,
        variant: str = DEFAULT_VARIANT,
        eval_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(judge)
        if variant not in ("ipi", "original"):
            raise ValueError(f"variant must be 'ipi' or 'original', got {variant!r}")
        self.prompt_num = prompt_num
        self.variant    = variant
        self.eval_mode  = eval_mode
        self.seed       = seed

    @classmethod
    def requires_local_target(cls) -> bool:
        return False

    # ------------------------------------------------------------------
    # The attack
    # ------------------------------------------------------------------

    def single_attack(self, target: Victim, instance: Instance,
                      verbose: bool = False) -> AttackDataset:
        """Build the demonstration transcript, send it once, report it."""
        goal = instance.query or ""
        target_str, eval_mode = resolve_attack_target(instance, self.eval_mode)

        query = VictimQuery(instance, target)
        candidate = Instance(
            id=f"ica-{instance.id}",
            query=goal,
            jailbreak_prompt=build_ica_injection(
                goal, prompt_num=self.prompt_num, variant=self.variant, seed=self.seed),
            reference_responses=[target_str] if target_str else [],
            attack_attrs={"target_str": target_str, "attack_eval_mode": eval_mode},
        )

        dataset = query(AttackDataset([candidate]))
        EvaluatorIPISuccess(mode=eval_mode)(dataset)

        success = bool(candidate.eval_results and candidate.eval_results[-1])
        if verbose:
            log.info("[ica] scenario=%s success=%s", instance.id, success)

        return self.report(
            candidate, query,
            success=success,
            score=self._grade(candidate, goal),
            prompt_num=self.prompt_num,
            variant=self.variant,
        )

    def _grade(self, candidate: Instance, goal: str) -> Optional[int]:
        """
        The reported 1-10 score: the judge's if there is one, else the binary verdict.

        ``None`` lets ``build_result`` fall back to ``eval_results[-1]`` mapped through
        ``Evaluator.as_score`` (10 or 1).
        """
        if self.judge is None:
            return None
        try:
            return self.judge.score(
                candidate.attack_attrs.get("injection", "")
                or candidate.jailbreak_prompt or "",
                candidate.target_responses[-1] if candidate.target_responses else "",
                attacker_goal=goal,
            )
        except Exception as exc:
            log.debug("[ICA] judge.score failed: %s", exc)
            return None

    def __repr__(self) -> str:
        return (f"ICAAttacker(prompt_num={self.prompt_num}, variant={self.variant!r}, "
                f"eval_mode={self.eval_mode!r})")
