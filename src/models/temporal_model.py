"""
Phase 3 - temporal behaviour features.

When someone posts is a genuinely strong cross-platform identity signal in
the real world (sleep schedule, timezone, work rhythm) and needs no private
data whatsoever - it is derivable from public timestamps alone. The original
proposal never considered it.

Hours are circular (23:00 and 01:00 are two hours apart, not twenty-two), so
comparisons use circular statistics rather than plain arithmetic on hour
numbers.

Coverage is deliberately uneven: Telegram and LinkedIn accounts have 0-2
posts in the generated world, so these features are often None or based on
very few samples. `min_posts` guards against reading a confident rhythm out
of a single timestamp.
"""
import numpy as np

MIN_POSTS = 3


def posting_hours(account: dict) -> list[int]:
    posts = account.get("posts")
    if posts is None:
        return []
    hours = []
    for p in posts:
        if not isinstance(p, dict):
            continue
        ts = p.get("timestamp")
        if not ts:
            continue
        try:
            hours.append(int(str(ts)[11:13]))
        except (ValueError, IndexError):
            continue
    return hours


def hour_profile(account: dict) -> dict | None:
    """Circular mean/concentration plus a normalised 24-bin histogram."""
    hours = posting_hours(account)
    if len(hours) < MIN_POSTS:
        return None

    angles = np.array(hours, dtype=np.float64) * (2 * np.pi / 24)
    mean_vec = np.array([np.mean(np.cos(angles)), np.mean(np.sin(angles))])
    resultant = float(np.linalg.norm(mean_vec))  # 0 = uniform, 1 = perfectly concentrated
    circular_mean = float(np.arctan2(mean_vec[1], mean_vec[0]) % (2 * np.pi))

    hist = np.bincount(np.array(hours), minlength=24).astype(np.float32)
    hist /= hist.sum()

    return {"circular_mean": circular_mean, "concentration": resultant, "histogram": hist}


def _circular_distance(a: float, b: float) -> float:
    """Shortest angular distance in radians, in [0, pi]."""
    d = abs(a - b) % (2 * np.pi)
    return float(min(d, 2 * np.pi - d))


def temporal_similarity(profile_a: dict | None, profile_b: dict | None) -> dict:
    if profile_a is None or profile_b is None:
        return {"temporal_peak_match": None, "temporal_histogram_cosine": None}

    # 1.0 when peak hours coincide, 0.0 when 12 hours apart
    peak_match = 1.0 - _circular_distance(profile_a["circular_mean"], profile_b["circular_mean"]) / np.pi

    ha, hb = profile_a["histogram"], profile_b["histogram"]
    denom = float(np.linalg.norm(ha) * np.linalg.norm(hb))
    hist_cosine = float(np.dot(ha, hb) / denom) if denom > 0 else None

    return {"temporal_peak_match": float(peak_match), "temporal_histogram_cosine": hist_cosine}


def extract(a: dict, b: dict, profiles: dict) -> dict:
    return temporal_similarity(profiles.get(a["record_id"]), profiles.get(b["record_id"]))
