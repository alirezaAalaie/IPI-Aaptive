"""
ipi.datasets — the carrier object and the benchmark.

Mirrors EasyJailbreak's ``easyjailbreak/datasets/``:

    Instance        one attack candidate / scenario — the object every component reads
    AttackDataset   an ordered collection of Instances — what every component takes
                    and returns (upstream's JailbreakDataset)

plus our benchmark:

    DualVerifiableDataset / load_dual_verifiable

This replaces the old ``ipi/dataset.py``, whose ``IPIScenario`` was an immutable input
record with no room for search state, and whose five other loaders (Manual, AgentDojo,
Bipia, Hijack) were removed — DualVerifiable is the benchmark we run.
"""
from .instance import Instance
from .attack_dataset import AttackDataset
from .dual_verifiable import DualVerifiableDataset, load_dual_verifiable

__all__ = [
    "Instance",
    "AttackDataset",
    "DualVerifiableDataset",
    "load_dual_verifiable",
]
