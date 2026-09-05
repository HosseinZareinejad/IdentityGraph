import random
import re

from hazm import Normalizer

# ---------------------------------------------------------------------------
# Persian -> Latin (Finglish) transliteration.
#
# Not linguistically perfect, but deterministic and shared between the
# synthetic-data generator (data/generate_synthetic_world.py, which uses it to
# derive usernames from real names) and the Phase-3 metadata matcher (which
# uses it to score username<->full-name similarity). Keeping one shared table
# means "generate" and "match" agree on what a name transliterates to.
# ---------------------------------------------------------------------------
_FA_LATIN_MAP = {
    "آ": "a", "ا": "a", "ب": "b", "پ": "p", "ت": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ذ": "z",
    "ر": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s",
    "ض": "z", "ط": "t", "ظ": "z", "ع": "", "غ": "gh", "ف": "f",
    "ق": "gh", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n",
    "و": "v", "ه": "h", "ی": "y", "ء": "", "ئ": "y", "ة": "h",
    " ": "", "‌": "",  # space and ZWNJ both collapse (usernames have no spaces)
}

# A handful of well-known compound first names and the informal nickname(s)
# people commonly go by. Deliberately conservative / high-confidence only —
# used both to add realistic nickname noise when generating data and as a
# real signal the metadata matcher can exploit (rather than inventing rules
# only one side of the pipeline knows about).
NICKNAME_MAP = {
    "محمدرضا": ["رضا", "محمد"],
    "علیرضا": ["علی", "رضا"],
    "حسینعلی": ["حسین"],
    "محمدحسین": ["محمد", "حسین"],
    "محمدجواد": ["جواد", "محمد"],
    "علی‌اکبر": ["اکبر", "علی"],
    "علیاکبر": ["اکبر", "علی"],
    "محمدعلی": ["علی", "محمد"],
    "امیرحسین": ["امیر", "حسین"],
    "امیرمحمد": ["امیر", "محمد"],
    "محمدامین": ["امین", "محمد"],
    "غلامرضا": ["رضا"],
    "عبدالرضا": ["رضا"],
    "سیدمحمد": ["محمد"],
    "فاطمه": ["فاطی"],
    "محمدمهدی": ["مهدی", "محمد"],
}


# Homophone-equivalence classes, collapsed to the first member of each class.
# Used only to build a blocking key - never for display or scoring. Every
# letter within a group is pronounced
# identically in Persian (unlike Arabic, where they differ), so the choice
# between them is orthographic convention that ordinary people get wrong.
#
# ض/ظ and ح were missing here until Phase 6 measured the key against 7.8k real
# Wikidata names: substituting a homophone into a real name changed its key
# 77% of the time, i.e. the blocking path silently missed most of the very
# confusions it exists to absorb. Adding them takes variant survival to 100%
# while collisions across 7,793 distinct real names stay at exactly one - so
# the recall is free.
_PHONETIC_CLASSES = [
    ("ز", "ذ", "ض", "ظ"),
    ("س", "ص", "ث"),
    ("ت", "ط"),
    ("ق", "غ"),
    ("ه", "ح"),
]
_PHONETIC_MAP = {ch: cls[0] for cls in _PHONETIC_CLASSES for ch in cls}

_phonetic_normalizer = Normalizer()


def compute_phonetic_key(name: str) -> str:
    """Collapse homophone letters + strip spaces, for exact-match blocking.

    Lives here rather than in etl/clean.py because the Phase 5 registry
    mapper needs the same key for registry records, which never pass through
    the ETL. One implementation, so the two sides can't drift.
    """
    if not name:
        return ""
    normalized = _phonetic_normalizer.normalize(name)
    collapsed = "".join(_PHONETIC_MAP.get(ch, ch) for ch in normalized)
    return collapsed.replace(" ", "").replace("‌", "")


def canonicalize_phone(phone):
    """+98/0-prefix -> a single canonical +98... form, for blocking only."""
    if phone is None or (isinstance(phone, float) and phone != phone):  # NaN
        return None
    digits = "".join(ch for ch in str(phone) if ch.isdigit() or ch == "+")
    if digits.startswith("+98"):
        return digits
    if digits.startswith("0098"):
        return "+98" + digits[4:]
    if digits.startswith("0"):
        return "+98" + digits[1:]
    return digits


def transliterate(text: str) -> str:
    """Deterministic Persian -> Latin (Finglish) transliteration, lowercase,
    with no separators (suitable for turning a name into a username stem)."""
    if not text:
        return ""
    out = []
    for ch in text:
        out.append(_FA_LATIN_MAP.get(ch, ch if ch.isascii() else ""))
    return "".join(out).lower()


# ---------------------------------------------------------------------------
# Vowelled romanisation.
#
# transliterate() above emits only the letters Persian script actually writes,
# so محمدرضا becomes "mhmdrza". No human writes their handle that way - they
# write "mohammadreza" - because Persian omits short vowels and the writer
# supplies them. Phase 6 measured this against 7.8k real (Persian, English)
# name pairs from Wikidata: the vowel-less form scores 0.785 mean Jaro-Winkler
# against human romanisations, the syllable-aware one below scores 0.846, and
# the share landing above 0.85 goes from 24% to 53%.
#
# It is used to GENERATE realistic synthetic handles, not to match them. The
# matcher can't use it, because reconstructing the same vowels a particular
# person chose is exactly the information the script doesn't carry - which is
# why the matching side instead compares consonant skeletons (latin_skeleton).
# ---------------------------------------------------------------------------
_ROMAN_CONSONANTS = {
    "ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h",
    "خ": "kh", "د": "d", "ذ": "z", "ر": "r", "ز": "z", "ژ": "zh", "س": "s",
    "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "غ": "gh", "ف": "f",
    "ق": "gh", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ه": "h",
}
# Letters that carry no sound of their own but do open a syllable.
_ROMAN_CARRIERS = {"ع", "ء", "أ", "إ", "ؤ", "ئ"}
_ROMAN_LONG = {"ا", "آ", "و", "ی", "ة"}
_SHORT_VOWELS = ["a", "e", "o"]


def _romanize_word(word: str, rng) -> str:
    """Persian syllables are CV(C)(C), so a short vowel belongs after a
    consonant only when that consonant OPENS a syllable. Tracking whether the
    current syllable already has a nucleus is what distinguishes "mahmoud"
    from "mahomod"."""
    chars = [c for c in word if c in _ROMAN_CONSONANTS or c in _ROMAN_LONG or c in _ROMAN_CARRIERS]
    out, nucleus = [], False

    for i, ch in enumerate(chars):
        nxt = chars[i + 1] if i + 1 < len(chars) else None

        if ch in ("ا", "آ", "ة"):
            out.append("h" if ch == "ة" else "a")
            nucleus = True
        elif ch == "و":
            if i == 0 or nucleus:          # consonantal v (ورزش, دیوار)
                out.append("v")
                nucleus = False
            else:
                out.append(rng.choice(["ou", "u", "o"]))
                nucleus = True
        elif ch == "ی":
            if i == 0 or nucleus:          # consonantal y
                out.append("y")
                nucleus = False
            else:
                out.append(rng.choice(["i", "i", "ee"]))
                nucleus = True
        elif ch in _ROMAN_CARRIERS:
            if not nucleus:
                out.append(rng.choice(_SHORT_VOWELS))
                nucleus = True
        elif nxt is None and not nucleus and out:
            # word-final consonant closing an empty syllable: the vowel goes
            # BEFORE it ("sadegh", "najaf"), not after
            out.append(rng.choice(_SHORT_VOWELS))
            out.append(_ROMAN_CONSONANTS[ch])
            nucleus = False
        else:
            out.append(_ROMAN_CONSONANTS[ch])
            if not nucleus and nxt is not None and nxt not in _ROMAN_LONG:
                out.append(rng.choice(_SHORT_VOWELS))
                nucleus = True
            elif nucleus:
                nucleus = False

    return re.sub(r"([aeiou])\1+", r"\1", "".join(out))


def romanize(text: str, rng=None) -> str:
    """Persian -> plausible Finglish, with short vowels supplied."""
    if not text:
        return ""
    rng = rng or random
    return "".join(_romanize_word(w, rng) for w in str(text).split() if w).lower()


_LATIN_VOWELS = str.maketrans("", "", "aeiouy")


def latin_skeleton(text: str) -> str:
    """Latin consonant skeleton: lowercase, letters only, vowels removed, runs
    of the same letter collapsed.

    Two romanisations of one Persian name disagree almost exclusively about two
    things the script never recorded:

      SHORT VOWELS - "hosseini" / "hoseyni" / "husayni" are one name.
      GEMINATION   - Persian marks a doubled consonant with a diacritic nobody
                     types, so "mohammadreza" and "mohamadreza" are equally
                     ordinary spellings of محمدرضا, and transliterate() (which
                     can only see one م) produces the single-consonant form.
                     Without collapsing runs, "mhmmdrz" and "mhmdrz" are not
                     even substrings of each other.

    Dropping both leaves what the two spellings genuinely share.

    Measured on the 7.8k real Wikidata name pairs, ranking one Persian name
    against all 7,812 real English labels: top-1 goes 66.1% (literal) -> 86.3%
    (vowels dropped) -> 90.0% (runs collapsed too), and MRR 0.732 -> 0.890 ->
    0.916.
    """
    stripped = re.sub(r"[^a-z]", "", (text or "").lower()).translate(_LATIN_VOWELS)
    return re.sub(r"(.)\1+", r"\1", stripped)


def get_nickname_variants(first_name: str) -> list[str]:
    """Return known informal nickname(s) for a compound first name, or []."""
    return NICKNAME_MAP.get(first_name, [])


class TextNormalizer:
    def __init__(self):
        self.normalizer = Normalizer()

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        return self.normalizer.normalize(str(text).strip())

    def build_composite_text(self, name: str, city: str, bio: str) -> str:
        n_name = self.normalize(name) or "نامشخص"
        n_city = self.normalize(city) or "نامشخص"
        n_bio = self.normalize(bio) or "بدون بیوگرافی"
        return f"[نام: {n_name}] [شهر: {n_city}] [بیوگرافی: {n_bio}]"

# Global instance for easy import
normalizer = TextNormalizer()
