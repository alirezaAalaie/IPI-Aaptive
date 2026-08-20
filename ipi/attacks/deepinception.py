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

Composition
-----------
One template, one query, no search — the whole algorithm is three lines inside
``single_attack``::

    query   = VictimQuery(instance, target)         # owns the call and the count
    dataset = query(AttackDataset([candidate]))
    EvaluatorIPISuccess(mode=eval_mode)(dataset)    # ground truth, not the judge

IPI adaptation vs original
--------------------------
  {query}:   the attacker's embedded instruction / injection goal (not a harmful request).
  delivery:  the filled template is injected through the data channel via
             ``harness.VictimQuery`` (untrusted document), not sent as a user turn.
  success:   resolved from the scenario (``resolve_attack_target``) to match the dataset's
             own ``attack_eval_mode``.
  score:     with no ``judge=``, the reported 1-10 score is the binary verdict mapped
             through ``Evaluator.as_score`` (10 / 1) rather than the hard 0 the deleted
             ``run_deepinception`` returned. ``AttackEvaluator`` recomputes success
             regardless — the judge only annotates.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..attacker import StaticAttacker
from ..datasets import AttackDataset, Instance
from ..harness import VictimQuery
from ..metrics import Evaluator, EvaluatorIPISuccess, resolve_attack_target
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


class DeepInceptionAttacker(StaticAttacker):
    """
    DeepInception attacker (single-query, template-based, API-compatible).

    From: Li et al. (2023), "DeepInception: Hypnotize Large Language Model to Be
    Jailbreaker".

    Args:
        judge:            Optional guidance Evaluator (ipi.metrics). Annotates the
                          reported candidate with a 1-10 score; it never decides
                          success.
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

    # ------------------------------------------------------------------
    # The attack
    # ------------------------------------------------------------------

    def single_attack(self, target: Victim, instance: Instance,
                      verbose: bool = False) -> AttackDataset:
        """Fill the nested-fiction template with the goal, send it once, report it."""
        goal = instance.query or ""
        target_str, eval_mode = resolve_attack_target(instance, self.eval_mode)

        query = VictimQuery(instance, target)
        candidate = Instance(
            id=f"deepinception-{instance.id}",
            query=goal,
            jailbreak_prompt=build_deepinception_injection(
                goal, self.scene, self.character_number, self.layer_number),
            reference_responses=[target_str] if target_str else [],
            attack_attrs={"target_str": target_str, "attack_eval_mode": eval_mode},
        )

        dataset = query(AttackDataset([candidate]))
        EvaluatorIPISuccess(mode=eval_mode)(dataset)

        success = bool(candidate.eval_results and candidate.eval_results[-1])
        if verbose:
            log.info("[deepinception] scenario=%s success=%s", instance.id, success)

        return self.report(
            candidate, query,
            success=success,
            score=self._grade(candidate, goal),
            scene=self.scene,
            layers=self.layer_number,
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
            log.debug("[DeepInception] judge.score failed: %s", exc)
            return None

    def __repr__(self) -> str:
        return (
            f"DeepInceptionAttacker(scene={self.scene!r}, "
            f"characters={self.character_number}, layers={self.layer_number})"
        )
