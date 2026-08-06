"""
SecAlign Defense Module (Compatibility wrapper re-exporting from `ipi.defenses.secalign`).
"""
from .secalign.defense import SecAlignDefense
from .secalign.dataset import generate_secalign_preference_data, load_paper_datasets, format_with_other_delimiters
from .secalign.trainer import train_secalign

__all__ = [
    "SecAlignDefense",
    "generate_secalign_preference_data",
    "load_paper_datasets",
    "format_with_other_delimiters",
    "train_secalign",
]
