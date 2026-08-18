"""
Rule mutations — deterministic operators. No model, no network (except ``Translate``).

Upstream's ``mutation/rule/``. Four groups:

    encoding      Base64 (3 framings) · Rot13 · Leetspeak · Disemvowel ·
                  Combination1/2/3 — the payload is encoded, and a wrapper prompt tells
                  the victim to decode and comply
    CodeChameleon Reverse · OddEven · Length · BinaryTree — the payload is *scrambled*
                  and the wrapper hands the victim the decryption function source
    search        SentenceCrossOver (AutoDAN's GA) · ReplaceWordsWithSynonyms (its HGA)
    Translate     the payload in another language

Why these matter for IPI, not just jailbreaks
---------------------------------------------
Upstream uses them to get past a *safety* refusal. In our setting the barrier is
instruction provenance, and these operators attack a different layer: an injection that
arrives base64-encoded or word-scrambled is invisible to any defense that pattern-matches
the data channel for imperative text, while an instruction-tuned victim still decodes and
follows it. They compose with the structured defenses too — ``recursive_filter`` strips
``FILTERED_TOKENS`` from the data channel, and an encoded payload has none of them.

Wrapper prompts
---------------
Each operator attaches a ``jailbreak_prompt`` when the instance has none (upstream
behaviour). Those strings live in the seed registry under ``mutation.<Name>`` — verbatim,
and checked by ``scripts/check_seed_fidelity.py`` — rather than inline here, because the
registry is the single home for every fixed string an attack uses. The CodeChameleon
decryption-function sources are there too, as ``mutation.CodeChameleon<Scheme>``.

Not ported: ``AsciiExpert`` / ``CaesarExpert`` / ``MorseExpert`` / ``SelfDefineCipher``
(~1200 lines of cipher tables and system prompts that exist only for the Cipher recipe),
``MJPChoices`` (the MJP recipe), ``Artificial`` (needs upstream's ``Jailbroken`` seed
pool) and ``Inception`` (our ``attacks/deepinception.py`` is the same thing). The locked
decision is *no new attack recipes*, and an operator with no recipe to use it is dead
code — if a Cipher row is ever wanted, those system prompts belong in the seed registry.
"""
from __future__ import annotations

import logging
import base64 as _base64
import codecs
import json
import random
import re
from typing import Optional

from ..datasets import Instance
from ..seed import PLACEHOLDER, SeedTemplate
from .base import MutationBase

__all__ = [
    "Base64", "Base64InputOnly", "Base64Raw", "Rot13", "Leetspeak", "Disemvowel",
    "Combination1", "Combination2", "Combination3",
    "CodeChameleon", "Reverse", "OddEven", "Length", "BinaryTree",
    "SentenceCrossOver", "ReplaceWordsWithSynonyms", "Translate",
    "ENCODING_MUTATORS", "CODECHAMELEON_MUTATORS",
]


def _wrapper(method: str) -> str:
    """The verbatim ``jailbreak_prompt`` this operator attaches, from the registry."""
    return SeedTemplate().new_seeds(
        prompt_usage="mutation", method_list=[method], variant="original")[0]


class _RuleMutation(MutationBase):
    """A deterministic operator whose wrapper prompt comes from the registry."""

    #: Registry key under ``mutation``. None = attach the bare ``{query}`` wrapper.
    wrapper_method: Optional[str] = None

    def __init__(self, attr_name: Optional[str] = None):
        super().__init__(attr_name)
        self.default_jailbreak_prompt = (
            _wrapper(self.wrapper_method) if self.wrapper_method else PLACEHOLDER)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class Base64(_RuleMutation):
    """Base64-encode the payload; wrapper asks the victim to respond to it."""

    wrapper_method = "Base64"

    def _mutate(self, text: str, **kwargs) -> str:
        return _base64.b64encode(text.encode()).decode()


class Base64InputOnly(Base64):
    """Base64-encode; wrapper asks for the *response* in plain English."""

    wrapper_method = "Base64InputOnly"


class Base64Raw(Base64):
    """Base64-encode with no wrapper at all — the bare encoded string."""

    wrapper_method = None


class Rot13(_RuleMutation):
    """ROT13 the payload; wrapper tells the victim not to print the decoded message."""

    wrapper_method = "Rot13"

    def _mutate(self, text: str, **kwargs) -> str:
        return codecs.encode(text, "rot_13")


class Leetspeak(_RuleMutation):
    """Substitute leet characters (a->@, e->3, i->!, o->0, are->r, be->b)."""

    _LEET = {"a": "@", "e": "3", "i": "!", "o": "0", "are": "r", "be": "b"}

    def _mutate(self, text: str, **kwargs) -> str:
        for key, value in self._LEET.items():
            text = text.replace(key, value)
        return text


class Disemvowel(_RuleMutation):
    """Drop every vowel."""

    def _mutate(self, text: str, **kwargs) -> str:
        return "".join(c for c in text if c not in "aeiouAEIOU")


class Combination1(Base64):
    """Base64 + a forced compliance prefix and four output rules."""

    wrapper_method = "Combination1"


class Combination2(Base64):
    """Combination1 plus: no punctuation, no long words, never say "the"."""

    wrapper_method = "Combination2"


class Combination3(Base64):
    """Combination2 plus: answer in the style of a Wikipedia article."""

    wrapper_method = "Combination3"


def ENCODING_MUTATORS() -> list[MutationBase]:
    """The Jailbroken encoding family, in upstream's registration order."""
    return [
        Base64(), Base64InputOnly(), Base64Raw(),
        Disemvowel(), Leetspeak(), Rot13(),
        Combination1(), Combination2(), Combination3(),
    ]


# ---------------------------------------------------------------------------
# CodeChameleon — scramble the payload, hand over the decoder
# ---------------------------------------------------------------------------

class CodeChameleon(_RuleMutation):
    """
    Base for the four CodeChameleon schemes.

    The payload is scrambled by ``_mutate``; the wrapper is the shared ``ProblemSolver``
    template, with ``{decryption_function}`` filled in with this scheme's decoder source
    so the victim can reconstruct the instruction itself.

    Deviation from upstream: ``BinaryTree`` and ``Length`` encrypt to a Python ``dict`` /
    ``list`` and upstream interpolates it with ``str()``, which emits single-quoted Python
    repr — not JSON, so the decryption function it just handed over cannot parse it. We
    ``json.dumps`` instead. The decoder source is unchanged.
    """

    wrapper_method = "CodeChameleon"

    #: Registry key of this scheme's decryption function source.
    decryption_method: str = ""

    def __init__(self, attr_name: Optional[str] = None):
        super().__init__(attr_name)
        self.decryption_function = _wrapper(self.decryption_method)
        self.default_jailbreak_prompt = self.default_jailbreak_prompt.replace(
            "{decryption_function}", self.decryption_function)

    def _mutate(self, text: str, **kwargs) -> str:
        raise NotImplementedError


class Reverse(CodeChameleon):
    """Reverse the word order."""

    decryption_method = "CodeChameleonReverse"

    def _mutate(self, text: str, **kwargs) -> str:
        return " ".join(text.split(" ")[::-1])


class OddEven(CodeChameleon):
    """Odd-position words first, then even-position words."""

    decryption_method = "CodeChameleonOddEven"

    def _mutate(self, text: str, **kwargs) -> str:
        words = text.split()
        return " ".join(words[::2] + words[1::2])


class Length(CodeChameleon):
    """Sort words by length, carrying each word's original index."""

    decryption_method = "CodeChameleonLength"

    def _mutate(self, text: str, **kwargs) -> str:
        words = text.split()
        ordered = sorted(enumerate(words), key=lambda pair: len(pair[1]))
        return json.dumps([{word: index} for index, word in ordered], ensure_ascii=False)


class BinaryTree(CodeChameleon):
    """Build a balanced binary tree over the words; in-order traversal restores them."""

    decryption_method = "CodeChameleonBinaryTree"

    def _mutate(self, text: str, **kwargs) -> str:
        words = text.split()

        def build(start: int, end: int):
            if start > end:
                return None
            mid = (start + end) // 2
            return {
                "value": words[mid],
                "left": build(start, mid - 1),
                "right": build(mid + 1, end),
            }

        return json.dumps(build(0, len(words) - 1), ensure_ascii=False)


def CODECHAMELEON_MUTATORS() -> list[MutationBase]:
    """The four CodeChameleon encryption schemes."""
    return [Reverse(), OddEven(), Length(), BinaryTree()]


# ---------------------------------------------------------------------------
# Search operators (AutoDAN)
# ---------------------------------------------------------------------------

class SentenceCrossOver(MutationBase):
    """
    Interleave the sentences of two texts at ``num_points`` random break points.

    AutoDAN's GA crossover — deterministic, no model, unlike ``generation.CrossOver``
    which asks an LLM to blend two templates. Upstream calls both ``CrossOver``.

    1-to-2: returns both children, as upstream does. The partner comes from
    ``other_instance``/``other`` if given, else from ``seed_pool``.
    """

    attr_name = "jailbreak_prompt"

    def __init__(self, num_points: int = 5, seed_pool: Optional[list[str]] = None,
                 attr_name: Optional[str] = None, seed: Optional[int] = None):
        super().__init__(attr_name)
        self.num_points = num_points
        self.seed_pool = list(seed_pool or [])
        self._rng = random.Random(seed)

    def crossover(self, str1: str, str2: str, num_points: Optional[int] = None
                  ) -> tuple[str, str]:
        """Swap sentence runs between two texts. Verbatim from upstream."""
        rng = self._rng
        num_points = self.num_points if num_points is None else num_points
        sentences1 = [s for s in re.split(r"(?<=[.!?])\s+", str1) if s]
        sentences2 = [s for s in re.split(r"(?<=[.!?])\s+", str2) if s]

        max_swaps = min(len(sentences1), len(sentences2)) - 1
        num_swaps = min(num_points, max_swaps)
        if num_swaps >= max_swaps:
            return str1, str2

        swap_indices = sorted(rng.sample(range(1, max_swaps), num_swaps))
        new_str1: list[str] = []
        new_str2: list[str] = []
        last_swap = 0
        for swap in swap_indices:
            if rng.choice([True, False]):
                new_str1.extend(sentences1[last_swap:swap])
                new_str2.extend(sentences2[last_swap:swap])
            else:
                new_str1.extend(sentences2[last_swap:swap])
                new_str2.extend(sentences1[last_swap:swap])
            last_swap = swap
        if rng.choice([True, False]):
            new_str1.extend(sentences1[last_swap:])
            new_str2.extend(sentences2[last_swap:])
        else:
            new_str1.extend(sentences2[last_swap:])
            new_str2.extend(sentences1[last_swap:])
        return " ".join(new_str1), " ".join(new_str2)

    def _mutate(self, text: str, other: Optional[str] = None, **kwargs) -> str:
        return self.crossover(text, other or "")[0]

    def _get_mutated_instance(self, instance: Instance, other_instance=None,
                              **kwargs) -> list[Instance]:
        text = getattr(instance, self.attr_name, "") or ""
        if other_instance is not None:
            other = getattr(other_instance, self.attr_name, "") or ""
        elif self.seed_pool:
            other = self._rng.choice(self.seed_pool)
        else:
            return [self.new_child(instance)]

        child1_text, child2_text = self.crossover(text, other)
        children = []
        for new_text in (child1_text, child2_text):
            child = self.new_child(instance)
            setattr(child, self.attr_name, new_text)
            children.append(child)
        return children


log = logging.getLogger(__name__)

#: Warn once per process, not once per candidate — this is called hundreds of times a
#: generation.
_NLTK_WARNED = False


def _warn_nltk_missing(what: str, exc: BaseException) -> None:
    """
    Say loudly that a word-level operator has degraded to a pass-through.

    AutoDAN-HGA's only mutation between GA steps is synonym replacement. Without nltk
    it returns its input unchanged, so with the default ``hga_period=5`` **80 of every
    100 generations do nothing at all** — the search looks like it is running, the loss
    barely moves, and the best candidate is still a pristine seed template. That is
    exactly the kind of silent no-op the repo exists to avoid, so it warns rather than
    logging at debug.
    """
    global _NLTK_WARNED
    if _NLTK_WARNED:
        return
    _NLTK_WARNED = True
    log.warning(
        "[%s] nltk is unavailable (%s) — word-level mutation is a PASS-THROUGH for the "
        "rest of this run. AutoDAN-HGA does nothing on its non-GA generations. Fix with: "
        "pip install nltk && python -m nltk.downloader stopwords wordnet punkt punkt_tab",
        what, exc)


class ReplaceWordsWithSynonyms(MutationBase):
    """
    Replace words with synonyms drawn from AutoDAN-HGA's momentum word dictionary.

    A word is swapped for a synonym with probability proportional to that synonym's
    accumulated score, so words that appeared in high-fitness candidates spread through
    the population. Needs ``nltk`` (wordnet + punkt), imported lazily — ``import ipi``
    must not require it. With no dictionary, or no nltk, the text passes through
    unchanged rather than raising mid-search.
    """

    attr_name = "jailbreak_prompt"

    def __init__(self, word_dict: Optional[dict] = None, attr_name: Optional[str] = None,
                 seed: Optional[int] = None):
        super().__init__(attr_name)
        self.word_dict = dict(word_dict or {})
        self._rng = random.Random(seed)

    def update(self, word_dict: Optional[dict] = None) -> None:
        """Refresh the momentum dictionary between generations."""
        self.word_dict = dict(word_dict or {})

    def _mutate(self, text: str, **kwargs) -> str:
        if not self.word_dict:
            return text
        try:
            import nltk
            from nltk.corpus import wordnet
        except ImportError as exc:
            _warn_nltk_missing("ReplaceWordsWithSynonyms", exc)
            return text

        try:
            words = nltk.word_tokenize(text)
        except LookupError as exc:   # nltk installed but corpora not downloaded
            _warn_nltk_missing("ReplaceWordsWithSynonyms", exc)
            return text

        for word in words:
            synonyms = [
                lemma.name()
                for syn in wordnet.synsets(word)
                for lemma in syn.lemmas()
            ]
            scored = [s for s in synonyms if s in self.word_dict]
            if not scored:
                continue
            total = sum(self.word_dict[s] for s in scored)
            if total <= 0:
                continue
            for synonym in scored:
                if self._rng.random() < self.word_dict[synonym] / total:
                    text = text.replace(word, synonym, 1)
                    break
        return text


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

class Translate(MutationBase):
    """
    Translate the payload into another language via the keyless Google endpoint.

    Args:
        language:   One of the nine target languages the Multilingual paper uses, plus
                    ``en``.
        translator: ``callable(text, target_lang) -> str`` to translate without network.
                    Default hits ``translate.googleapis.com``.

    Note:
        ``default_jailbreak_prompt`` is upstream's **AIM** jailbreak wrapper, kept
        verbatim in the registry for a paper-faithful row. Our ``attacks/multilingual.py``
        deliberately does *not* use it — AIM is a direct-chat persona device, and for IPI
        the translated instruction *is* the injected content. Set
        ``default_jailbreak_prompt = None`` (or leave the instance's own prompt set) for
        the IPI framing.
    """

    _LANGUAGES = {
        "en": "English", "zh-CN": "Chinese", "it": "Italian", "vi": "Vietnamese",
        "ar": "Arabic", "ko": "Korean", "th": "Thai", "bn": "Bengali",
        "sw": "Swahili", "jv": "Javanese",
    }

    def __init__(self, language: str = "en", translator=None,
                 attr_name: Optional[str] = None):
        super().__init__(attr_name)
        if language not in self._LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Choose from {sorted(self._LANGUAGES)}")
        self.language = language
        self.lang = self._LANGUAGES[language]
        self.translator = translator
        self.default_jailbreak_prompt = _wrapper("Translate")

    def translate(self, text: str, src_lang: str = "auto") -> str:
        if self.translator is not None:
            return self.translator(text, self.language)
        import urllib.parse
        import urllib.request
        url = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={src_lang}&tl={self.language}&dt=t"
            f"&q={urllib.parse.quote(text)}"
        )
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        return "".join(segment[0] for segment in data[0])

    def _mutate(self, text: str, **kwargs) -> str:
        return self.translate(text)

    def _get_mutated_instance(self, instance: Instance, **kwargs) -> list[Instance]:
        children = super()._get_mutated_instance(instance, **kwargs)
        for child in children:
            child.lang = self.lang
            child.origin_query = getattr(instance, self.attr_name, "")
            child.translated_query = getattr(child, self.attr_name, "")
        return children
