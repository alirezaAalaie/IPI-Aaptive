"""
ipi.mutation — the operators that turn one attack candidate into others.

Mirrors EasyJailbreak's ``easyjailbreak/mutation/``. An operator takes an
``AttackDataset``, rewrites one field of each ``Instance``, and returns the children with
their lineage recorded — which is what lets a selector run a tree search over them.

    generation.py   needs a model: the operator prompts an LLM to rewrite the text
    rule.py         deterministic: encodings, scrambles, sentence crossover, synonyms

Upstream keeps one class per file under ``generation/`` and ``rule/``; we keep the same
two-way split as two modules. What matters is the taxonomy — whether an operator costs an
LLM call — not the file count.

    from ipi.mutation import GPTFUZZER_MUTATORS, RENELLM_MUTATORS, Base64, Rot13

Every instruction prompt is verbatim from upstream, and the wrapper prompts the rule
operators attach live in the seed registry under ``mutation.<Name>``, checked by
``scripts/check_seed_fidelity.py``.

Not here yet: ``gradient/``. Upstream's ``MutationTokenGradient`` is written against its
``WhiteBoxModelBase`` / ``model_utils.encode_trace`` stack, which we deliberately do not
have — our ``Victim`` contract replaces it, and ``attacks/gcg.py`` already implements the
same token-gradient step against it. The operator gets extracted from *our* GCG when that
recipe migrates in Phase H; porting upstream's would mean porting their model layer.
"""
from .base import MutationBase, LLMMutation, LLMCallable
from .generation import (
    Expand, Shorten, Rephrase, GenerateSimilar, CrossOver, GPTFUZZER_MUTATORS,
    AlterSentenceStructure, ChangeStyle, InsertMeaninglessCharacters,
    MisspellSensitiveWords, Translation, RENELLM_MUTATORS,
    AutoPayloadSplitting, AutoObfuscation, HistoricalInsight,
)
from .rule import (
    Base64, Base64InputOnly, Base64Raw, Rot13, Leetspeak, Disemvowel,
    Combination1, Combination2, Combination3, ENCODING_MUTATORS,
    CodeChameleon, Reverse, OddEven, Length, BinaryTree, CODECHAMELEON_MUTATORS,
    SentenceCrossOver, ReplaceWordsWithSynonyms, Translate,
)

__all__ = [
    # Base
    "MutationBase", "LLMMutation", "LLMCallable",
    # Generation — GPTFuzzer
    "Expand", "Shorten", "Rephrase", "GenerateSimilar", "CrossOver", "GPTFUZZER_MUTATORS",
    # Generation — ReNeLLM
    "AlterSentenceStructure", "ChangeStyle", "InsertMeaninglessCharacters",
    "MisspellSensitiveWords", "Translation", "RENELLM_MUTATORS",
    # Generation — Jailbroken / PAIR
    "AutoPayloadSplitting", "AutoObfuscation", "HistoricalInsight",
    # Rule — encoding
    "Base64", "Base64InputOnly", "Base64Raw", "Rot13", "Leetspeak", "Disemvowel",
    "Combination1", "Combination2", "Combination3", "ENCODING_MUTATORS",
    # Rule — CodeChameleon
    "CodeChameleon", "Reverse", "OddEven", "Length", "BinaryTree",
    "CODECHAMELEON_MUTATORS",
    # Rule — search / translation
    "SentenceCrossOver", "ReplaceWordsWithSynonyms", "Translate",
]
