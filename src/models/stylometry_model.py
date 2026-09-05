"""
Phase 3 - stylometry (authorship) features.

Rationale: semantic embeddings capture WHAT someone writes about; two
different people who both post about football look alike. Stylometry captures
HOW they write - punctuation habits, emoji rate, code-switching into
Finglish, sentence length, character n-gram habits - which is far more
identity-bearing and works on short text. The literature the project's own
report cites (Goga et al.) rests on exactly this "behavioural leakage".

HONESTY CAVEAT - read before trusting any number this module produces on
synthetic data: data/generate_synthetic_world.py injects style using three
explicit per-person parameters (emoji_rate, finglish_rate, punctuation_style).
Three of the features below measure those same three parameters, so on this
dataset the module is partly grading its own homework and WILL look better
than it would on real text. The character n-gram and length features are less
circular, but not independent either.

The honest validation is authorship attribution on the real Persian tweet
corpus (same-author vs different-author pairs), which Phase 0 set up as a
data source and Phase 6 is scheduled to run. Until that runs, treat
stylometry's ablation contribution here as an upper bound, not an estimate.
"""
import re

import numpy as np

# Common Persian function words - grammatical glue, chosen because function
# word frequencies are the classic authorship signal (content-independent).
FUNCTION_WORDS = [
    "از", "به", "با", "که", "را", "در", "این", "آن", "و", "یا", "تا", "هم",
    "برای", "اما", "اگر", "چون", "ولی", "بر", "هر", "همه", "خیلی", "باید",
    "شد", "بود", "است", "می", "من", "تو", "او", "ما", "شما", "خود",
]

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF❤️]"
)
_LATIN_RE = re.compile(r"[A-Za-z]{2,}")


def account_text(account: dict) -> str:
    """All of an account's free text: bio plus post bodies."""
    parts = []
    if isinstance(account.get("bio"), str):
        parts.append(account["bio"])
    posts = account.get("posts")
    if posts is not None:
        for p in posts:
            text = p.get("text") if isinstance(p, dict) else None
            if text:
                parts.append(text)
    return " ".join(parts)


def style_profile(account: dict) -> dict | None:
    """Per-account style vector. None when there simply isn't enough text to
    characterise a style (a bio-less, post-less Telegram account)."""
    text = account_text(account)
    if len(text.strip()) < 15:
        return None

    tokens = text.split()
    n_tokens = max(len(tokens), 1)
    n_chars = max(len(text), 1)

    fw_counts = np.array([sum(1 for t in tokens if t.strip(".,!؟،") == w) for w in FUNCTION_WORDS],
                         dtype=np.float32)
    fw_profile = fw_counts / n_tokens

    return {
        "emoji_rate": len(_EMOJI_RE.findall(text)) / n_chars,
        "latin_rate": sum(len(m) for m in _LATIN_RE.findall(text)) / n_chars,
        "ellipsis_rate": text.count("...") / max(n_tokens / 10, 1),
        "exclaim_rate": text.count("!") / max(n_tokens / 10, 1),
        "question_rate": text.count("?") / max(n_tokens / 10, 1),
        "avg_token_len": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        "function_words": fw_profile,
    }


def _scalar_similarity(x: float, y: float, scale: float) -> float:
    """1.0 when identical, decaying with absolute difference."""
    return float(np.exp(-abs(x - y) / scale))


def style_similarity(profile_a: dict | None, profile_b: dict | None) -> dict:
    if profile_a is None or profile_b is None:
        return {"style_punctuation": None, "style_emoji": None,
                "style_finglish": None, "style_function_words": None}

    fw_a, fw_b = profile_a["function_words"], profile_b["function_words"]
    denom = float(np.linalg.norm(fw_a) * np.linalg.norm(fw_b))
    fw_cosine = float(np.dot(fw_a, fw_b) / denom) if denom > 0 else None

    punctuation = np.mean([
        _scalar_similarity(profile_a["ellipsis_rate"], profile_b["ellipsis_rate"], 0.5),
        _scalar_similarity(profile_a["exclaim_rate"], profile_b["exclaim_rate"], 0.5),
        _scalar_similarity(profile_a["question_rate"], profile_b["question_rate"], 0.5),
    ])

    return {
        "style_punctuation": float(punctuation),
        "style_emoji": _scalar_similarity(profile_a["emoji_rate"], profile_b["emoji_rate"], 0.02),
        "style_finglish": _scalar_similarity(profile_a["latin_rate"], profile_b["latin_rate"], 0.05),
        "style_function_words": fw_cosine,
    }


def extract(a: dict, b: dict, profiles: dict) -> dict:
    return style_similarity(profiles.get(a["record_id"]), profiles.get(b["record_id"]))
