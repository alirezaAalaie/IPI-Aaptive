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
  - The demonstration pool is AdvBench (goal, target) pairs — the same harmful-behavior
    corpus the paper uses — served from the seed registry (``seeds.advbench_pairs``).
  - The transcript is ``User:{goal}\nAssistant:{target}`` repeated ``prompt_num`` times,
    then the attacker goal.

IPI adaptation vs original
--------------------------
  The original ICA sends the demonstrations as **real alternating chat turns** to a raw
  chat model — a *direct* prompt-injection setting. For **indirect** prompt injection the
  payload must travel through the data channel, so the whole transcript (demos + goal) is
  emitted as one injected-document string and delivered by ``make_scenario_target_fn`` as
  untrusted content. The demonstrations themselves are unchanged (real AdvBench pairs),
  which is what makes this dataset-agnostic: it no longer assumes the goal is a tool call.

  success: resolved from the scenario (``resolve_attack_target``) so it matches the
           dataset's own ``attack_eval_mode`` (e.g. DualVerifiableDataset uses
           ``startswith`` / ``contains`` against a literal target string).

Report of differences from the reference is in docs/easyjailbreak-audit.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..attacker import StaticAttacker
from ..evaluator import check_ipi_success
from ..judges import Judge
from ..victim import Victim
from .seeds import advbench_pairs

log = logging.getLogger(__name__)

DEFAULT_PROMPT_NUM = 5   # 5-shot, matching the reference ICA template


def build_ica_injection(
    goal: str,
    prompt_num: int = DEFAULT_PROMPT_NUM,
    demos: Optional[list[tuple[str, str]]] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Build the ICA injection: ``prompt_num`` AdvBench compliance demonstrations followed
    by the attacker goal, formatted exactly like the reference template
    (``User:{q}\\nAssistant:{a}`` ... ``User:{goal}``).

    Args:
        goal:       The attacker instruction placed after the demonstrations.
        prompt_num: Number of in-context demonstrations (paper default: 5).
        demos:      Optional (goal, target) pairs to use instead of the AdvBench sample.
        seed:       RNG seed for sampling the AdvBench pairs (reproducibility).
    """
    if demos is None:
        demos = advbench_pairs(n=prompt_num, seed=seed)
    demos = demos[:prompt_num]

    lines: list[str] = []
    for demo_goal, demo_target in demos:
        lines.append(f"User:{demo_goal}")
        lines.append(f"Assistant:{demo_target}")
    lines.append(f"User:{goal}")
    return "\n".join(lines)


@dataclass
class ICAResult:
    success: bool
    score: int
    injection: str
    target_response: str
    n_queries: int          # always 1
    goal: str
    prompt_num: int = DEFAULT_PROMPT_NUM
    trace: list[dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ICAResult(success={self.success}, score={self.score}, prompt_num={self.prompt_num})"


def run_ica(
    goal: str,
    target_fn: Callable[[str], str],
    judge: Optional[Judge] = None,
    prompt_num: int = DEFAULT_PROMPT_NUM,
    demos: Optional[list[tuple[str, str]]] = None,
    eval_mode: str = "contains",
    target_str: str = "",
    seed: Optional[int] = None,
) -> ICAResult:
    """
    Run the single-query In-Context Attack.

    Args:
        goal:       Attacker injection goal.
        target_fn:  Callable(injection: str) -> response: str.
        judge:      Optional Judge for scoring.
        prompt_num: Number of AdvBench compliance demonstrations. Default 5.
        demos:      Optional custom (goal, target) demonstration pairs.
        eval_mode:  check_ipi_success mode (usually resolved from the scenario).
        target_str: Expected success string (falls back to goal).
        seed:       RNG seed for demo sampling.

    Returns:
        ICAResult (n_queries=1).
    """
    injection = build_ica_injection(goal, prompt_num=prompt_num, demos=demos, seed=seed)

    try:
        response = target_fn(injection)
    except Exception as exc:
        log.warning("[ICA] target_fn raised: %s", exc)
        response = ""

    check_str = target_str or goal
    success = check_ipi_success(response, check_str, eval_mode)

    score = 0
    if judge is not None:
        try:
            score = judge.score(injection, response, attacker_goal=goal)
        except Exception as exc:
            log.debug("[ICA] judge.score failed: %s", exc)

    return ICAResult(
        success=success,
        score=score,
        injection=injection,
        target_response=response,
        n_queries=1,
        goal=goal,
        prompt_num=prompt_num,
        trace=[{"injection": injection, "response": response, "success": success}],
    )


class ICAAttacker(StaticAttacker):
    """
    In-Context Attack (ICA) — single-query, AdvBench-demonstration-based, API-compatible.

    From: Wei et al. (2023), "Jailbreak and Guard Aligned Language Models with Only Few
    In-Context Demonstrations".

    Args:
        judge:      Optional Judge for scoring.
        prompt_num: Number of AdvBench compliance demonstrations. Default 5 (paper).
        eval_mode:  check_ipi_success mode. Default None → auto-detect from the scenario's
                    ``attack_eval_mode`` (e.g. DualVerifiableDataset).
        seed:       RNG seed for reproducible demo sampling.
    """

    _ATTACK_NAME = "ica"

    def __init__(
        self,
        judge: Optional[Judge] = None,
        prompt_num: int = DEFAULT_PROMPT_NUM,
        eval_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(judge)
        self.prompt_num = prompt_num
        self.eval_mode  = eval_mode
        self.seed       = seed

    @classmethod
    def requires_local_target(cls) -> bool:
        return False

    def run_scenario(self, target: Victim, scenario, verbose: bool = False):
        from ..evaluator import ScenarioResult, make_scenario_target_fn, resolve_attack_target
        target_fn = make_scenario_target_fn(scenario, target)
        target_str, eval_mode = resolve_attack_target(scenario, self.eval_mode)

        r = run_ica(
            goal=scenario.injection_goal,
            target_fn=target_fn,
            judge=self.judge,
            prompt_num=self.prompt_num,
            eval_mode=eval_mode,
            target_str=target_str,
            seed=self.seed,
        )

        if verbose:
            log.info("[ica] scenario=%s success=%s score=%d",
                     scenario.id, r.success, r.score)

        return ScenarioResult(
            scenario_id=scenario.id,
            goal=scenario.injection_goal,
            success=r.success,
            score=r.score,
            injection=r.injection,
            target_response=r.target_response,
            n_queries=r.n_queries,
            attack=self._ATTACK_NAME,
            extra={"prompt_num": self.prompt_num},
        )

    def __repr__(self) -> str:
        return f"ICAAttacker(prompt_num={self.prompt_num}, eval_mode={self.eval_mode!r})"
