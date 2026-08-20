"""
GPTFuzzer adapted for Indirect Prompt Injection.

Original paper: Yu et al. (2023), "GPTFUZZER: Red Teaming Large Language Models with
Auto-Generated Jailbreak Prompts"
arXiv: https://arxiv.org/abs/2309.10253
Source: https://github.com/sherdencooper/GPTFuzz

Mechanism
---------
A black-box fuzzing loop over a pool of seed templates:
  1. Select a seed with MCTS — walk down the mutation tree from a root, balancing a
     seed's past reward against how rarely it has been tried.
  2. Mutate it with one of five LLM operators — Expand, Shorten, Rephrase,
     GenerateSimilar, CrossOver — each keeping the ``{query}`` placeholder.
  3. Render the mutant with the goal, query the target, evaluate, add the mutant to the
     pool as a new node, and back-propagate its reward up the selected path.
Repeat until a query budget is hit; stop early on success.

This recipe is **pure wiring**: the pool is an ``AttackDataset`` of ``Instance``s, the
operators come from ``ipi.mutation``, selection from ``ipi.selector``, success from
``ipi.metrics``, and the seeds from ``ipi.seed``. There is no attack-local candidate type,
prompt text or selection rule left.

Fidelity
--------
  seeds:      the 77 GPTFuzzer initial seed templates, verbatim (``attack.Gptfuzzer.original``).
  mutators:   the five operators with their upstream prompts (``ipi/mutation/``).
  selection:  ``MCTSExploreSelectPolicy`` — the reference's own policy, formulas verbatim.
              (Until the carrier existed this recipe used flat UCB1, because MCTS needs
              per-node ``index``/``visited_num``/``level``/``children``.)
  reward:     binary, as upstream — its RoBERTa judge emits a jailbreak *label*, and
              ``Instance.num_jailbreak`` sums those labels. We substitute the dataset's own
              ground truth (``EvaluatorIPISuccess``), which is stronger than a classifier:
              dual-verifiable scenarios know exactly what success looks like.
  delivery:   the rendered template is injected through the data channel via
              ``harness.make_target_fn`` (untrusted content), not a user turn.
  success:    resolved from the scenario (``resolve_attack_target``) to match the dataset's
              own ``attack_eval_mode``.

IPI adaptations vs original
---------------------------
  reward source: upstream trains a RoBERTa classifier on jailbreak/no-jailbreak pairs.
      We have deterministic ground truth per scenario, so ``EvaluatorIPISuccess`` replaces
      it. A guidance evaluator (``judge=``) is **optional and annotative**: it grades the
      candidate that ends up being reported, so the row carries a 1-10 score rather than
      a bare 10/1, and it does not drive selection. Before the MCTS port it did
      (normalised to [0,1]), which was a workaround for not shipping the classifier, not
      a property of the algorithm. It is now one judge call per scenario rather than one
      per candidate — the per-candidate scores only ever fed a trace that stopped at the
      ``ScenarioResult`` boundary.
  budget:  the reference fuzzes to a fixed iteration count over a whole question set; we
      budget by target queries per scenario, which is the scarce resource here.

Composition
-----------
One round is::

    parent  = self.selector.select()[0]          # MCTSExploreSelectPolicy
    child   = mutator(AttackDataset([parent]))   # one of GPTFUZZER_MUTATORS
    batch   = query(AttackDataset([child]))      # harness.VictimQuery (budgeted)
    self.evaluator(batch)                        # EvaluatorIPISuccess
    self.selector.update(tried)                  # back-propagate up the path

The query budget lives on ``VictimQuery``, so the loop condition is
``while not query.exhausted`` rather than a hand-kept counter.

Report of differences from the reference is in docs/easyjailbreak-audit.md.
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Union

from ..attacker import AdaptiveAttacker, as_attacker_llm
from ..datasets import AttackDataset, Instance
from ..harness import VictimQuery
from ..llm_unified import APILLM
from ..metrics import Evaluator, EvaluatorIPISuccess, resolve_attack_target
from ..mutation import GPTFUZZER_MUTATORS
from ..seed import SeedTemplate, render
from ..selector import MCTSExploreSelectPolicy
from ..victim import Victim

log = logging.getLogger(__name__)

__all__ = ["GPTFuzzerAttacker"]


def _load_seeds() -> list[str]:
    """The 77 verbatim GPTFuzzer seed templates from the registry."""
    return SeedTemplate().new_seeds(
        prompt_usage="attack", method_list=["Gptfuzzer"], variant="original",
    )


class GPTFuzzerAttacker(AdaptiveAttacker):
    """
    Budgeted seed-template fuzzing with MCTS selection.

    Args:
        attacker_llm:   LLM (or model string) driving the five mutation operators.
        judge:          Optional guidance Evaluator. Grades the reported candidate so
                        the row carries a 1-10 score; the *search* reward is binary
                        success against the scenario's ground truth either way.
        max_queries:    Victim-query budget, the stopping criterion. Default 40.
        energy:         Mutants generated per selected seed each round. Default 1.
        eval_mode:      ``check_ipi_success`` mode. Default None → read the scenario's
                        own ``attack_eval_mode``. Pass a string only to force a
                        deliberate deviation.
        seed_templates: Override the seed pool (default: the 77 upstream seeds).
        seed:           RNG seed, for reproducibility.
    """

    _ATTACK_NAME = "gptfuzzer"

    def __init__(
        self,
        attacker_llm: Union[str, APILLM],
        judge: Optional[Evaluator] = None,
        max_queries: int = 40,
        energy: int = 1,
        eval_mode: Optional[str] = None,
        seed_templates: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(judge)
        self.attacker_llm = as_attacker_llm(attacker_llm)
        self.max_queries = max_queries
        self.energy = energy
        self.eval_mode = eval_mode
        self.seed = seed
        self.templates = seed_templates or _load_seeds()
        # Operators rewrite the *template*, not the goal; CrossOver draws from the pool.
        self.mutators = GPTFUZZER_MUTATORS(
            self.attacker_llm, seed_pool=self.templates, attr_name="jailbreak_prompt")

    # ------------------------------------------------------------------
    # The attack
    # ------------------------------------------------------------------

    def single_attack(self, target: Victim, instance: Instance,
                      verbose: bool = False) -> AttackDataset:
        """
        Fuzz until the query budget runs out, or a mutant succeeds.

        The selector, the mutation pool and the evaluator are the components; the loop
        only decides *when* to call them. Every mutant is registered with the policy
        (which assigns its reward slot) and reached through a mutation operator (which
        records the parent/child edge and the level MCTS descends).
        """
        goal = instance.query or ""
        target_str, eval_mode = resolve_attack_target(instance, self.eval_mode)
        rng = random.Random(self.seed)

        pool = self._seed_pool(goal, target_str, eval_mode)
        selector = MCTSExploreSelectPolicy(
            dataset=pool, initial_prompt_pool=AttackDataset(list(pool)),
            questions=1, seed=self.seed)
        evaluator = EvaluatorIPISuccess(mode=eval_mode)
        # The template still carries {query}; render it at query time so the reported
        # attack_str is the string the victim saw, not the template behind it.
        query = VictimQuery(
            instance, target, budget=self.max_queries,
            render=lambda candidate: render(candidate.jailbreak_prompt, goal))

        best: Optional[Instance] = None
        succeeded = False
        iterations = 0

        while not query.exhausted and not succeeded:
            iterations += 1
            parent = selector.select()[0]
            tried = AttackDataset([])

            for _ in range(self.energy):
                if query.exhausted:
                    break
                child = rng.choice(self.mutators)(AttackDataset([parent]))[0]
                # A mutant has not been tried yet; it inherited the parent's transcript
                # through copy(), which HistoricalInsight wants and we do not.
                child.target_responses = []
                child.eval_results = []
                selector.register(child)

                batch = query(AttackDataset([child]))
                if not len(batch):
                    break
                evaluator(batch)
                tried.add(child)
                best = self.keep_best(best, batch)

                if child.eval_results and child.eval_results[-1]:
                    succeeded = True
                    if verbose:
                        log.info("[GPTFuzzer] success at iteration=%d after %d queries",
                                 iterations, query.n_queries)
                    break

            selector.update(tried)                    # back-propagate up the path

        return self.report(
            best, query,
            success=succeeded,
            score=self._grade(best, goal),
            iterations=iterations,
            best_template=best.jailbreak_prompt if best is not None else "",
            # How far MCTS descended the mutation tree. 1 means it only ever re-rolled
            # roots — which is all flat UCB1 could do before the carrier existed.
            max_level=max((node.level for node in selector.dataset), default=0),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seed_pool(self, goal: str, target_str: str, eval_mode: str) -> AttackDataset:
        """The initial population: one root ``Instance`` per seed template."""
        return AttackDataset([
            Instance(
                id=f"seed-{i}",
                query=goal,
                jailbreak_prompt=template,
                reference_responses=[target_str],
                attack_attrs={"target_str": target_str, "attack_eval_mode": eval_mode},
            )
            for i, template in enumerate(self.templates)
        ])

    def _grade(self, best: Optional[Instance], goal: str) -> Optional[int]:
        """
        The reported 1-10 score: the judge's if there is one, else the binary verdict.

        ``None`` lets ``build_result`` fall back to ``eval_results[-1]`` — the binary
        success mapped through ``Evaluator.as_score`` (10 or 1).
        """
        if best is None or self.judge is None:
            return None
        try:
            return self.judge.score(
                best.attack_attrs.get("injection", "") or best.jailbreak_prompt or "",
                best.target_responses[-1] if best.target_responses else "",
                attacker_goal=goal,
            )
        except Exception as exc:
            log.debug("[GPTFuzzer] judge.score failed: %s", exc)
            return None

    def __repr__(self) -> str:
        return (f"GPTFuzzerAttacker(attacker={self.attacker_llm.model_name!r}, "
                f"max_queries={self.max_queries}, energy={self.energy})")
