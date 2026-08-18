"""
Selection policies — which candidate gets the next query.

Ports of ``easyjailbreak/selector/*``. Every formula below is upstream's, unchanged:

    RandomSelectPolicy      uniform draw
    RoundRobinSelectPolicy  cycle the pool in order
    UCBSelectPolicy         UCB1 over the flat pool
    EXP3SelectPolicy        exponential weights, for an adversarial (non-stationary) pool
    MCTSExploreSelectPolicy tree search down the mutation lineage — GPTFuzzer's own
    SelectBasedOnScores     keep the top-``tree_width`` by score — TAP's pruning

Which one a recipe wants
------------------------
The three bandits (UCB, EXP3, MCTS) reward with ``Instance.num_jailbreak`` and therefore
need a **binary** evaluator; ``SelectBasedOnScores`` reads ``eval_results[-1]`` as a
magnitude and wants a 1-10 ``*GetScore``. See ``base.py``.

``MCTSExploreSelectPolicy`` is the one that was previously unportable: it descends
``Instance.children`` and discounts by ``Instance.level``, neither of which existed
before the carrier. GPTFuzzer's flat UCB1 was a consequence of that, not a design choice.
"""
from __future__ import annotations

import random
from typing import Optional, Union

import numpy as np

from ..datasets import AttackDataset, Instance
from .base import SelectPolicy

__all__ = [
    "RandomSelectPolicy",
    "RoundRobinSelectPolicy",
    "UCBSelectPolicy",
    "EXP3SelectPolicy",
    "MCTSExploreSelectPolicy",
    "SelectBasedOnScores",
]


class RandomSelectPolicy(SelectPolicy):
    """Uniformly random candidate. The baseline every other policy has to beat."""

    def __init__(self, dataset: Optional[AttackDataset] = None, seed: Optional[int] = None):
        super().__init__(dataset, seed)
        self._rng = random.Random(seed)

    def select(self) -> AttackDataset:
        instance = self._rng.choice(list(self.dataset))
        instance.visited_num += 1
        return AttackDataset([instance])


class RoundRobinSelectPolicy(SelectPolicy):
    """
    Cycle through the pool in order.

    Note:
        ``update()`` steps the cursor *backwards*, so the same candidate is served again
        after a reported outcome. That is upstream's behaviour and it is deliberate —
        a recipe that calls ``update()`` every round stays on one candidate until it
        stops calling it. Kept verbatim; don't "fix" it without checking the recipe.
    """

    def __init__(self, dataset: Optional[AttackDataset] = None, seed: Optional[int] = None):
        super().__init__(dataset, seed)
        self.index: int = 0

    def select(self) -> AttackDataset:
        instance = self.dataset[self.index]
        instance.visited_num += 1
        self.index = (self.index + 1) % len(self.dataset)
        return AttackDataset([instance])

    def update(self, dataset: Optional[AttackDataset] = None):
        self.index = (self.index - 1 + len(self.dataset)) % len(self.dataset)


class UCBSelectPolicy(SelectPolicy):
    """
    UCB1 over the flat pool.

        score_i = reward_i / (visits_i + 1)
                  + explore_coeff * sqrt(2 * ln(step) / (visits_i + 1))

    Args:
        explore_coeff: Exploration weight. Higher = try neglected candidates more.
        dataset:       The candidate pool.
    """

    def __init__(self, explore_coeff: float = 1.0,
                 dataset: Optional[AttackDataset] = None, seed: Optional[int] = None):
        super().__init__(dataset, seed)
        self.step = 0
        self.last_choice_index: int = 0
        self.explore_coeff = explore_coeff
        self.rewards = [0.0 for _ in range(len(self.dataset))]

    def _grow(self, size: int):
        if size > len(self.rewards):
            self.rewards.extend([0.0] * (size - len(self.rewards)))

    def select(self) -> AttackDataset:
        self._grow(len(self.dataset))
        self.step += 1
        scores = np.zeros(len(self.dataset))
        for i, instance in enumerate(self.dataset):
            smooth_visited_num = instance.visited_num + 1
            scores[i] = (
                self.rewards[i] / smooth_visited_num
                + self.explore_coeff * np.sqrt(2 * np.log(self.step) / smooth_visited_num)
            )
        self.last_choice_index = int(np.argmax(scores))
        self.dataset[self.last_choice_index].visited_num += 1
        return AttackDataset([self.dataset[self.last_choice_index]])

    def update(self, dataset: AttackDataset):
        if not len(dataset):
            return
        succ_num = sum(instance.num_jailbreak for instance in dataset)
        self.rewards[self.last_choice_index] += succ_num / len(dataset)


class EXP3SelectPolicy(SelectPolicy):
    """
    Exponential-weight exploration/exploitation (EXP3).

        p_i  = (1 - gamma) * w_i / sum(w) + gamma / n
        r    = 1 - successes / len(batch)          # loss, not reward
        w_i *= exp(alpha * (-r / p_i) / n)

    Built for a pool whose payoffs shift under you — a victim that adapts, or a defense
    that starts filtering what worked last round.

    Args:
        energy: Carried from upstream, where it is stored and never read.
        gamma:  Exploration floor: every candidate keeps at least ``gamma / n`` mass.
        alpha:  Learning rate on the weight update.
    """

    def __init__(self, dataset: Optional[AttackDataset] = None, energy: float = 1.0,
                 gamma: float = 0.05, alpha: float = 25, seed: Optional[int] = None):
        super().__init__(dataset, seed)
        self.energy = energy
        self.gamma = gamma
        self.alpha = alpha
        self.last_choice_index: Optional[int] = None
        self._rng = np.random.default_rng(seed)
        self.initial()

    def initial(self):
        self.weights = [1.0 for _ in range(len(self.dataset))]
        self.probs = [0.0 for _ in range(len(self.dataset))]

    def _grow(self, size: int):
        if size > len(self.weights):
            self.weights.extend([1.0] * (size - len(self.weights)))
        if size > len(self.probs):
            self.probs.extend([0.0] * (size - len(self.probs)))

    def select(self) -> AttackDataset:
        self._grow(len(self.dataset))
        np_weights = np.array(self.weights)
        probs = ((1 - self.gamma) * np_weights / np_weights.sum()
                 + self.gamma / len(self.dataset))
        self.last_choice_index = int(self._rng.choice(len(self.dataset), p=probs))
        self.dataset[self.last_choice_index].visited_num += 1
        self.probs[self.last_choice_index] = probs[self.last_choice_index]
        return AttackDataset([self.dataset[self.last_choice_index]])

    def update(self, dataset: AttackDataset):
        if not len(dataset):
            return
        succ_num = sum(instance.num_jailbreak for instance in dataset)
        r = 1 - succ_num / len(dataset)
        x = -1 * r / self.probs[self.last_choice_index]
        self.weights[self.last_choice_index] *= np.exp(
            self.alpha * x / len(self.dataset))


class MCTSExploreSelectPolicy(SelectPolicy):
    """
    Monte-Carlo tree search over the mutation lineage — GPTFuzzer's own policy.

    Selection walks down from a root in the initial pool, at each step taking the child
    with the best UCB-style score, and stops early with probability ``alpha``:

        score = reward_i / (visits_i + 1)
                + ratio * sqrt(2 * ln(step) / (visits_i + 0.01))

    Back-propagation credits the whole selected path, discounted by how deep the chosen
    node sits — a success reachable from a shallow seed is worth more than one that took
    ten mutations to find:

        reward = successes / (n_questions * len(batch))
        reward_p += reward * max(beta, 1 - 0.1 * level_of_chosen_node)   for p in path

    Args:
        dataset:             The full pool (roots and every mutant added since).
        initial_prompt_pool: The roots selection starts from. Defaults to ``dataset``.
        questions:           The goals each candidate is tried against — only its
                             *length* is used, as the reward denominator. An int is
                             accepted directly.
        ratio:               Exploration weight.
        alpha:               Per-step probability of stopping the descent early.
        beta:                Floor on the depth discount.
    """

    def __init__(
        self,
        dataset: Optional[AttackDataset] = None,
        initial_prompt_pool: Optional[AttackDataset] = None,
        questions: Union[AttackDataset, int, None] = None,
        ratio: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.2,
        seed: Optional[int] = None,
    ):
        super().__init__(dataset, seed)
        self.initial_prompt_pool = (
            initial_prompt_pool if initial_prompt_pool is not None else self.dataset)
        self.n_questions = (
            questions if isinstance(questions, int)
            else (len(questions) if questions is not None else 1))
        self.step = 0
        self.select_path: list[Instance] = []
        self.last_choice_index: Optional[int] = None
        self.rewards: list[float] = []
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta
        self._rng = np.random.default_rng(seed)

    def _grow(self, size: int):
        if size > len(self.rewards):
            self.rewards.extend([0.0] * (size - len(self.rewards)))

    def _score(self, instance: Instance) -> float:
        return (
            self.rewards[instance.index] / (instance.visited_num + 1)
            + self.ratio * np.sqrt(2 * np.log(self.step) / (instance.visited_num + 0.01))
        )

    def select(self) -> AttackDataset:
        self.step += 1
        self._grow(len(self.dataset))
        self.select_path = []

        current = max(self.initial_prompt_pool, key=self._score)
        self.select_path.append(current)

        while len(current.children) > 0:
            if self._rng.random() < self.alpha:
                break
            current = max(current.children, key=self._score)
            self.select_path.append(current)

        for node in self.select_path:
            node.visited_num += 1
        self.last_choice_index = current.index
        return AttackDataset([current])

    def update(self, dataset: AttackDataset):
        if not len(dataset):
            return
        succ_num = sum(instance.num_jailbreak for instance in dataset)
        last_choice_node = self.dataset[self.last_choice_index]
        reward = succ_num / (self.n_questions * len(dataset))
        discount = max(self.beta, 1 - 0.1 * last_choice_node.level)
        for node in reversed(self.select_path):
            self.rewards[node.index] += reward * discount


class GeneticSelectPolicy(SelectPolicy):
    """
    AutoDAN's selection rule: keep the elites, breed the rest by roulette wheel.

    ``select()`` returns the whole next generation's *parents* — ``num_elites`` best
    candidates passed through untouched, then ``size - num_elites`` drawn with
    replacement by softmax roulette over fitness. A recipe then runs its crossover and
    mutation operators over the parents and puts the elites back in front.

    Fitness is read from ``Instance.eval_results[-1]`` and is **higher-is-better**. A
    loss-driven search (``ReferenceLossSelector`` writes ``instance._loss``) must negate
    first — upstream's ``autodan_sample_control`` opens with ``score_list = [-x for x in
    score_list]`` for exactly this reason, and getting the sign wrong inverts the search
    without erroring.

    Args:
        dataset:     Candidate pool. Also accepted per-call by ``select()``.
        num_elites:  Candidates carried over unchanged. Upstream is
                     ``max(1, int(batch_size * 0.05))``.
        if_softmax:  Softmax roulette (upstream's default) or score-proportional.
        seed:        RNG seed.
    """

    def __init__(self, dataset: Optional[AttackDataset] = None, num_elites: int = 1,
                 if_softmax: bool = True, seed: Optional[int] = None):
        super().__init__(dataset, seed)
        self.num_elites = num_elites
        self.if_softmax = if_softmax
        self._np_rng = np.random.default_rng(seed)

    @staticmethod
    def _fitness(instance: Instance) -> float:
        return float(instance.eval_results[-1]) if instance.eval_results else float("-inf")

    def elites(self, dataset: Optional[AttackDataset] = None) -> list[Instance]:
        """The ``num_elites`` fittest candidates, best first."""
        pool = list(dataset if dataset is not None else self.dataset)
        return sorted(pool, key=self._fitness, reverse=True)[:self.num_elites]

    def roulette(self, dataset: Optional[AttackDataset] = None,
                 n: Optional[int] = None) -> list[Instance]:
        """``n`` parents drawn with replacement, weighted by fitness."""
        pool = list(dataset if dataset is not None else self.dataset)
        if not pool:
            return []
        n = len(pool) - self.num_elites if n is None else n
        if n <= 0:
            return []

        scores = np.array([self._fitness(i) for i in pool], dtype=np.float64)
        # An -inf (unscored, or a degenerate candidate the loss selector gave up on)
        # would poison both branches; floor it to the worst finite score.
        finite = scores[np.isfinite(scores)]
        scores = np.where(np.isfinite(scores), scores,
                          finite.min() if finite.size else 0.0)

        if self.if_softmax:
            probs = np.exp(scores - scores.max())
        else:
            probs = scores - scores.min()      # shift to non-negative
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(len(pool), 1.0 / len(pool))

        idx = self._np_rng.choice(len(pool), size=n, p=probs, replace=True)
        return [pool[i] for i in idx]

    def select(self, dataset: Optional[AttackDataset] = None) -> AttackDataset:
        """Elites first, then the roulette-drawn parents — one full generation."""
        pool = AttackDataset(list(dataset if dataset is not None else self.dataset))
        return AttackDataset(self.elites(pool) + self.roulette(pool))

    def update(self, dataset: AttackDataset):
        """Genetic selection is stateless between generations — fitness is on the pool."""
        return


class SelectBasedOnScores(SelectPolicy):
    """
    Keep the ``tree_width`` highest-scoring candidates — TAP's pruning step.

    Reads ``eval_results[-1]`` as a magnitude, so it belongs with a 1-10 ``*GetScore``
    evaluator. Ties are broken by shuffling first, so a run does not systematically
    favour whichever branch happened to be generated first.

    Args:
        dataset:    Default pool, used when ``select()`` is called with no argument.
        tree_width: How many candidates survive.

    Deviation from upstream: when nothing scores above zero, upstream falls back to
    ``[list[0], list[1]]``, which raises ``IndexError`` on a single-candidate dataset —
    reachable on the last level of a narrow tree. We take ``[:2]``.
    """

    def __init__(self, dataset: Optional[AttackDataset] = None, tree_width: int = 10,
                 seed: Optional[int] = None):
        super().__init__(dataset, seed)
        self.tree_width = tree_width
        self._rng = random.Random(seed)

    def select(self, dataset: Optional[AttackDataset] = None) -> AttackDataset:
        pool = list(dataset if dataset is not None else self.dataset)
        if not pool:
            return AttackDataset([])

        self._rng.shuffle(pool)                       # random permutation within a score
        pool.sort(key=lambda i: i.eval_results[-1], reverse=True)

        width = min(self.tree_width, len(pool))
        truncated = [pool[i] for i in range(width) if pool[i].eval_results[-1] > 0]
        if not truncated:
            truncated = pool[:2]
        return AttackDataset(truncated)
