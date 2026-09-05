"""
Phase 3 - name & username comparison features.

Phase 2's retrieval measurement showed name-based paths carry the bulk of the
recoverable signal (name fuzzy ~69%, username fuzzy ~59%, phonetic key ~55%)
while text embeddings are supplementary (~20%). So this module gets the most
attention, not the text model.

Every function returns either a float in [0,1] or None. None means "no
evidence available" (a field was missing), NOT "evidence of a mismatch" -
the fusion model treats None as a neutral zero-weight level. The old
src/scorer.py conflated the two by scoring missing fields as 0.0, which
actively penalised sparse profiles.
"""
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import token_sort_ratio

from src.text_utils import NICKNAME_MAP, latin_skeleton, transliterate

TITLES = {"دکتر", "مهندس", "آقای", "خانم", "سید", "سیده", "استاد"}


def strip_titles(name: str) -> str:
    """Remove honorific prefixes the generator injects as noise (and which
    real Persian profiles genuinely carry)."""
    if not name:
        return ""
    tokens = [t for t in name.split() if t not in TITLES]
    return " ".join(tokens)


def strip_digits(text: str) -> str:
    return "".join(ch for ch in text if not ch.isdigit()) if text else ""


def name_similarity(name_a: str, name_b: str) -> float | None:
    """Jaro-Winkler on title-stripped names."""
    a, b = strip_titles(name_a or ""), strip_titles(name_b or "")
    if not a or not b:
        return None
    return JaroWinkler.normalized_similarity(a, b)


def name_token_sort_similarity(name_a: str, name_b: str) -> float | None:
    """Order-insensitive comparison - catches the first/last-name swap that is
    both a generator noise type and extremely common in real Iranian records
    ("رضایی محمد" vs "محمد رضایی")."""
    a, b = strip_titles(name_a or ""), strip_titles(name_b or "")
    if not a or not b:
        return None
    return token_sort_ratio(a, b) / 100.0


def phonetic_key_match(key_a: str, key_b: str) -> float | None:
    """Exact match on the homophone-collapsed key computed in ETL (handles
    ز/ذ, س/ص/ث, ت/ط confusions)."""
    if not key_a or not key_b:
        return None
    return 1.0 if key_a == key_b else 0.0


def nickname_compatible(name_a: str, name_b: str) -> float | None:
    """Is one name's first token a known informal form of the other's?
    (محمدرضا -> رضا). Returns None when neither name has a known nickname
    entry, so 'no dictionary coverage' isn't read as 'not compatible'."""
    a, b = strip_titles(name_a or ""), strip_titles(name_b or "")
    if not a or not b:
        return None
    first_a = a.split()[0] if a.split() else ""
    first_b = b.split()[0] if b.split() else ""
    if first_a == first_b:
        return None  # identical - no nickname evidence either way, name sim covers it

    a_variants = set(NICKNAME_MAP.get(first_a, []))
    b_variants = set(NICKNAME_MAP.get(first_b, []))
    if not a_variants and not b_variants:
        return None
    return 1.0 if (first_b in a_variants or first_a in b_variants) else 0.0


def username_similarity(user_a: str, user_b: str) -> float | None:
    """Handle-to-handle, scored in both the literal and the vowel-free space.

    Two handles built from the same Persian name usually differ only in the
    vowels their owner chose ("hosseini" / "hoseyni"), which the script never
    specified. Taking the better of the two readings costs nothing when the
    spellings already agree and recovers the pair when they don't.
    """
    if not user_a or not user_b:
        return None
    literal = JaroWinkler.normalized_similarity(user_a, user_b)
    skel_a, skel_b = latin_skeleton(user_a), latin_skeleton(user_b)
    if not skel_a or not skel_b:
        return literal
    return max(literal, JaroWinkler.normalized_similarity(skel_a, skel_b))


def username_similarity_nodigits(user_a: str, user_b: str) -> float | None:
    """Same, ignoring trailing digits ("ali.rezaei" vs "ali.rezaei87")."""
    a, b = strip_digits(user_a or ""), strip_digits(user_b or "")
    if not a or not b:
        return None
    return Levenshtein.normalized_similarity(a, b)


def username_vs_name(username: str, display_name: str) -> float | None:
    """Cross-field: does one account's username look like the OTHER account's
    real name transliterated? ("m.hosseini87" vs "محمد حسینی").

    The original src/scorer.py only ever compared username-to-username and
    name-to-name, so this signal was completely unused - despite being one of
    the strongest available when one platform shows a handle and another shows
    a legal name (exactly the Twitter<->LinkedIn case).

    The comparison happens in the consonant-skeleton space. transliterate()
    can only emit the letters Persian writes, so it produces "mhmdrza" while
    the handle its owner actually chose says "mohammadreza" - a Jaro-Winkler
    of 0.65 between two spellings of the same name. Dropping the vowels from
    both sides removes precisely the information the script never carried.
    Measured on 7,812 real Wikidata name pairs, ranking one Persian name
    against all of them, this takes top-1 from 66.1% to 86.3%.
    """
    if not username or not display_name:
        return None
    handle = strip_digits(username).replace(".", "").replace("_", "").replace("-", "")
    name_translit = transliterate(strip_titles(display_name))
    if not handle or not name_translit:
        return None
    literal = JaroWinkler.normalized_similarity(handle, name_translit)
    skel_h, skel_n = latin_skeleton(handle), latin_skeleton(name_translit)
    if not skel_h or not skel_n:
        return literal
    return max(literal, JaroWinkler.normalized_similarity(skel_h, skel_n))


def extract(a: dict, b: dict) -> dict:
    """All name/username features for one candidate pair."""
    return {
        "name_jw": name_similarity(a.get("display_name"), b.get("display_name")),
        "name_token_sort": name_token_sort_similarity(a.get("display_name"), b.get("display_name")),
        "name_phonetic": phonetic_key_match(a.get("name_phonetic_key"), b.get("name_phonetic_key")),
        "name_nickname": nickname_compatible(a.get("display_name"), b.get("display_name")),
        "username_jw": username_similarity(a.get("username"), b.get("username")),
        "username_nodigits": username_similarity_nodigits(a.get("username"), b.get("username")),
        # both directions, then keep the stronger - we don't know which side
        # carries the legal name and which carries the handle
        "username_vs_name": _best(
            username_vs_name(a.get("username"), b.get("display_name")),
            username_vs_name(b.get("username"), a.get("display_name")),
        ),
    }


def _best(x: float | None, y: float | None) -> float | None:
    vals = [v for v in (x, y) if v is not None]
    return max(vals) if vals else None
