"""
ipi.selector — which candidate gets the next query.

Mirrors EasyJailbreak's ``easyjailbreak/selector/``. A ``SelectPolicy`` owns a pool of
``Instance`` candidates and decides where to spend the next (expensive) victim query,
then folds the outcome back in:

    policy.select() -> AttackDataset
    policy.update(evaluated_dataset)

    RandomSelectPolicy       uniform — the baseline
    RoundRobinSelectPolicy   cycle in order
    UCBSelectPolicy          UCB1 over a flat pool
    EXP3SelectPolicy         exponential weights, for a pool whose payoffs shift
    MCTSExploreSelectPolicy  tree search down the mutation lineage (GPTFuzzer's own)
    SelectBasedOnScores      keep the top-k by score (TAP's pruning)
    ReferenceLossSelector    white-box: lowest teacher-forced loss on the target

The bandits reward with ``Instance.num_jailbreak`` and need a **binary** evaluator;
``SelectBasedOnScores`` reads ``eval_results[-1]`` as a magnitude and wants a 1-10
``*GetScore``. See ``base.py`` and ``ipi.metrics``.

    from ipi.selector import MCTSExploreSelectPolicy, SelectBasedOnScores
"""
from .base import SelectPolicy
from .policies import (
    RandomSelectPolicy,
    RoundRobinSelectPolicy,
    UCBSelectPolicy,
    EXP3SelectPolicy,
    MCTSExploreSelectPolicy,
    SelectBasedOnScores,
    GeneticSelectPolicy,
)
from .reference_loss import ReferenceLossSelector
from .token_loss import TokenLossSelector, ADV_IDS

__all__ = [
    "SelectPolicy",
    "RandomSelectPolicy",
    "RoundRobinSelectPolicy",
    "UCBSelectPolicy",
    "EXP3SelectPolicy",
    "MCTSExploreSelectPolicy",
    "SelectBasedOnScores",
    "GeneticSelectPolicy",
    "ReferenceLossSelector",
    "TokenLossSelector",
    "ADV_IDS",
]
