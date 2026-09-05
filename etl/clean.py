"""
Phase 2 - ETL step 1: cleaning & standardization.

Takes the raw per-platform dumps from Phase 1 (data/platform_dumps/*.parquet)
and produces cleaned, analysis-ready tables (data/cleaned/*.parquet).

What counts as "cleaning" here vs. what must NOT be touched:
  - Persian character-form unification (ي -> ی, ك -> ک, stray whitespace) is
    genuine cleanup: these are the same letter in different Unicode
    presentation forms, not real-world noise. hazm's Normalizer already does
    this correctly and is applied to every free-text field.
  - Things like the +98/0 phone-format difference, name typos, phonetic
    substitutions (ز/ذ, س/ص/ث, ت/ط), title prefixes, and name-order swaps are
    DELIBERATE noise injected in Phase 1 to simulate real duplicate records.
    Cleaning must not silently erase this signal - the matching models in
    Phase 3 are supposed to see through it. Where a canonical form is useful
    for blocking (e.g. phone), it is added as a SEPARATE derived column
    (phone_canonical) alongside the untouched raw value, exactly the way a
    real record-linkage system would (canonicalize for indexing, keep raw
    for scoring/audit) - this is standard practice, not "cheating" the noise
    away.
  - A phonetic key (name_phonetic_key) is likewise a separate derived column
    for blocking, not a replacement for display_name.
"""
import json
import os
import sys

import pandas as pd
from hazm import Normalizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.text_utils import transliterate  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATFORM_DUMP_DIR = os.path.join(BASE_DIR, "data", "platform_dumps")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
PLATFORMS = ["twitter", "instagram", "telegram", "linkedin"]

TEXT_FIELDS = ["display_name", "bio", "city", "job_title", "education"]

# Phonetic-equivalence classes (post hazm-normalization) used ONLY for a
# blocking key, never for display or scoring.
_PHONETIC_CLASSES = [
    ("ز", "ذ"),
    ("س", "ص", "ث"),
    ("ت", "ط"),
    ("ق", "غ"),
]
_PHONETIC_MAP = {ch: cls[0] for cls in _PHONETIC_CLASSES for ch in cls}

_normalizer = Normalizer()


def normalize_field(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = _normalizer.normalize(str(value).strip())
    return text if text else None


def compute_phonetic_key(name: str) -> str:
    """Collapse homophone letters + strip spaces, for exact-match blocking."""
    if not name:
        return ""
    normalized = _normalizer.normalize(name)
    collapsed = "".join(_PHONETIC_MAP.get(ch, ch) for ch in normalized)
    return collapsed.replace(" ", "").replace("‌", "")


def canonicalize_phone(phone):
    """+98/0-prefix -> a single canonical +98... form, for blocking only."""
    if phone is None or (isinstance(phone, float) and pd.isna(phone)):
        return None
    digits = "".join(ch for ch in str(phone) if ch.isdigit() or ch == "+")
    if digits.startswith("+98"):
        return digits
    if digits.startswith("0098"):
        return "+98" + digits[4:]
    if digits.startswith("0"):
        return "+98" + digits[1:]
    return digits


def parse_posts(posts_json: str) -> list[dict]:
    if not posts_json:
        return []
    try:
        return json.loads(posts_json)
    except (json.JSONDecodeError, TypeError):
        return []


def clean_platform_dump(platform: str) -> pd.DataFrame:
    src_path = os.path.join(PLATFORM_DUMP_DIR, f"{platform}.parquet")
    df = pd.read_parquet(src_path)

    for field in TEXT_FIELDS:
        df[field] = df[field].apply(normalize_field)

    df["posts"] = df["posts"].apply(parse_posts)
    df["posts_text_joined"] = df["posts"].apply(
        lambda posts: _normalizer.normalize(" ".join(p["text"] for p in posts[:5])) if posts else ""
    )
    df["num_posts"] = df["posts"].apply(len)

    df["name_phonetic_key"] = df["display_name"].apply(compute_phonetic_key)
    df["display_name_translit"] = df["display_name"].apply(transliterate)
    df["phone_canonical"] = df["phone_number"].apply(canonicalize_phone)
    df["email_normalized"] = df["email"].apply(lambda e: e.strip().lower() if isinstance(e, str) else None)

    return df


def run():
    os.makedirs(CLEANED_DIR, exist_ok=True)
    summary = {}
    for platform in PLATFORMS:
        df = clean_platform_dump(platform)
        out_path = os.path.join(CLEANED_DIR, f"{platform}.parquet")
        df.to_parquet(out_path, index=False)
        summary[platform] = len(df)
        print(f"  cleaned {platform}: {len(df)} rows -> {out_path}")
    return summary


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("Cleaning platform dumps...")
    run()
