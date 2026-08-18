"""
ipi.constraint — drop candidates before they cost a victim query.

Mirrors EasyJailbreak's ``easyjailbreak/constraint/``. A constraint runs after a mutation
and filters; it never rewrites (that is ``ipi.mutation``) and never scores for the record
(that is ``ipi.metrics``).

    constraint(dataset) -> dataset

    DeleteOffTopic        TAP's Phase-1 pruning — does this still ask for the goal?
    DeleteHarmLess        ReNeLLM's filter — is the rewritten payload still harmful?
                          Deliberately unused by our IPI ReNeLLM row; see filters.py
    PerplexityConstraint  keep only what would survive a perplexity defense

    from ipi.constraint import DeleteOffTopic, PerplexityConstraint
"""
from .base import ConstraintBase
from .filters import DeleteOffTopic, DeleteHarmLess
from .perplexity import PerplexityConstraint

__all__ = [
    "ConstraintBase",
    "DeleteOffTopic",
    "DeleteHarmLess",
    "PerplexityConstraint",
]
