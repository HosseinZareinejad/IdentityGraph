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


def transliterate(text: str) -> str:
    """Deterministic Persian -> Latin (Finglish) transliteration, lowercase,
    with no separators (suitable for turning a name into a username stem)."""
    if not text:
        return ""
    out = []
    for ch in text:
        out.append(_FA_LATIN_MAP.get(ch, ch if ch.isascii() else ""))
    return "".join(out).lower()


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
