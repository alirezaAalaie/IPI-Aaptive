"""
DeepInception adapted for Indirect Prompt Injection.

Original paper: Li et al. (2023), "DeepInception: Hypnotize Large Language Model
to Be Jailbreaker"
arXiv: https://arxiv.org/abs/2311.03191
Source: https://github.com/tmlr-group/DeepInception

Mechanism
---------
A single template asks the model to build a deeply *nested* fiction: layer 0 creates
layer 1, and so on; at each layer characters propose a step toward the goal, and the
final layer aggregates those steps into direct, actionable commands.

Fidelity
--------
The template is loaded **verbatim** from the seed registry
(``original/DeepInception``, copied from EasyJailbreak's seed_template.json). The three
knobs are applied exactly as the reference ``DeepInception`` recipe does — by
``str.replace`` on the seed ('science fiction' → scene, '4' → character_number,
'5' → layer_number) rather than by named format fields.

IPI adaptation vs original
--------------------------
  {query}:   the attacker's embedded instruction / injection goal (not a harmful request).
  delivery:  the filled template is injected through the data channel via
             ``make_scenario_target_fn`` (untrusted document), not sent as a user turn.
  success:   resolved from the scenario (``resolve_attack_target``) to match the dataset's
             own ``attack_eval_mode``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..attacker import StaticAttacker
from ..metrics import Evaluator, check_ipi_success
from ..victim import Victim
from ..seed import SeedTemplate, render

log = logging.getLogger(__name__)


def _load_template() -> str:
    """The verbatim DeepInception seed from the registry."""
    return SeedTemplate().new_seeds(
        prompt_usage="attack", method_list=["DeepInception"], variant="original",
    )[-1]


def build_deepinception_injection(
    goal: str,
    scene: str = "science fiction",
    character_number: int = 5,
    layer_number: int = 5,
) -> str:
    """
    Fill the verbatim DeepInception template with the attacker goal and knobs.

    Knobs are applied by string replacement, mirroring the reference recipe:
      'science fiction' -> scene,  '4' -> character_number,  '5' -> layer_number.

    Deviation from upstream: the reference applies these replacements *after*
    substituting {query}, which corrupts any goal containing '4' or '5'. We apply
    them to the template first, then substitute the goal — same template output for
    the default knobs, without mangling the payload.
    """
    tmpl = _load_template()
    if scene is not None:
        tmpl = tmpl.replace("science fiction", scene)
    if character_number is not None:
        tmpl = tmpl.replace("4", str(character_number))
    if layer_number is not None:
        tmpl = tmpl.replace("5", str(layer_number))
    return render(tmpl, goal)   # substitute {query} last


@dataclass
class DeepInceptionResult:
    success: bool
    score: int
    injection: str
    target_response: str
    n_queries: int          # always 1
    goal: str
    trace: list[dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"DeepInceptionResult(success={self.success}, score={self.score})"


def run_deepinception(
    goal: str,
    target_fn: Callable[[str], str],
    judge: Optional[Evaluator] = None,
    scene: str = "science fiction",
    character_number: int = 5,
    layer_number: int = 5,
    eval_mode: str = "contains",
    target_str: str = "",
) -> DeepInceptionResult:
    """
    Run the single-query DeepInception attack.

    Args:
        goal:             Attacker injection goal.
        target_fn:        Callable(injection: str) -> response: str.
        judge:            Optional guidance Evaluator (ipi.metrics) for scoring.
        scene:            Fictional setting. Default "science fiction".
        character_number: Characters per layer. Default 5.
        layer_number:     Nesting depth. Default 5.
        eval_mode:        check_ipi_success mode (usually resolved from the scenario).
        target_str:       Expected success string (falls back to goal).

    Returns:
        DeepInceptionResult (n_queries=1).
    """
    injection = build_deepinception_injection(goal, scene, character_number, layer_number)

    try:
        response = target_fn(injection)
    except Exception as exc:
        log.warning("[DeepInception] target_fn raised: %s", exc)
        response = ""

    check_str = target_str or goal
    success = check_ipi_success(response, check_str, eval_mode)

    score = 0
    if judge is not None:
        try:
            score = judge.score(injection, response, attacker_goal=goal)
        except Exception as exc:
            log.debug("[DeepInception] judge.score failed: %s", exc)

    return DeepInceptionResult(
        success=success,
        score=score,
        injection=injection,
        target_response=response,
        n_queries=1,
        goal=goal,
        trace=[{"injection": injection, "response": response, "success": success}],
    )


class DeepInceptionAttacker(StaticAttacker):
    """
    DeepInception attacker (single-query, template-based, API-compatible).

    From: Li et al. (2023), "DeepInception: Hypnotize Large Language Model to Be
    Jailbreaker".

    Args:
        judge:            Optional guidance Evaluator (ipi.metrics) for scoring.
        scene:            Fictional setting. Default "science fiction".
        character_number: Characters per layer. Default 5.
        layer_number:     Nesting depth. Default 5.
        eval_mode:        check_ipi_success mode. Default None → auto-detect from the
                          scenario's ``attack_eval_mode``.
    """

    _ATTACK_NAME = "deepinception"

    def __init__(
        self,
        judge: Optional[Evaluator] = None,
        scene: str = "science fiction",
        character_number: int = 5,
        layer_number: int = 5,
        eval_mode: Optional[str] = None,
    ):
        super().__init__(judge)
        self.scene            = scene
        self.character_number = character_number
        self.layer_number     = layer_number
        self.eval_mode        = eval_mode

    @classmethod
    def requires_local_target(cls) -> bool:
        return False

    def run_scenario(self, target: Victim, scenario, verbose: bool = False):
        from ..evaluator import make_scenario_target_fn
        from ..metrics import ScenarioResult, resolve_attack_target
        target_fn = make_scenario_target_fn(scenario, target)
        target_str, eval_mode = resolve_attack_target(scenario, self.eval_mode)

        r = run_deepinception(
            goal=scenario.injection_goal,
            target_fn=target_fn,
            judge=self.judge,
            scene=self.scene,
            character_number=self.character_number,
            layer_number=self.layer_number,
            eval_mode=eval_mode,
            target_str=target_str,
        )

        if verbose:
            log.info("[deepinception] scenario=%s success=%s score=%d",
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
            extra={"scene": self.scene, "layers": self.layer_number},
        )

    def __repr__(self) -> str:
        return (
            f"DeepInceptionAttacker(scene={self.scene!r}, "
            f"characters={self.character_number}, layers={self.layer_number})"
        )
