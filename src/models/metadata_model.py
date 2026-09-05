"""
Phase 3 - structured metadata comparison features.

Covers city, birth year, job/education, and the deterministic phone/email
paths. Per Phase 2's coverage matrix these fields are sparse and very
unevenly distributed across platforms (job/education only on LinkedIn, phone
mostly on Telegram, birth year almost nowhere), so almost every feature here
returns None most of the time. That is the normal case, not an error - which
is precisely why the fusion model must treat None as neutral.

Roadmap deviation: the roadmap called for a RandomForest trained on these
metadata features, feeding the ensemble. That would stack a black-box model
underneath the fusion layer and destroy the per-feature log-odds
decomposition that makes the final score explainable. Since the fusion model
(src/models/fusion_model.py) already learns a weight per comparison level
from labelled data, it subsumes the RandomForest's job with one trained model
instead of two - better calibrated and still interpretable.
"""

from rapidfuzz.distance import JaroWinkler


def city_match(city_a: str | None, city_b: str | None) -> float | None:
    if not city_a or not city_b:
        return None
    return JaroWinkler.normalized_similarity(str(city_a), str(city_b))


def birth_year_proximity(year_a, year_b) -> float | None:
    """1.0 for an exact match, decaying to 0 at a 10-year gap. The generator
    injects off-by-one and decade typos, so exact equality alone is too
    brittle."""
    if year_a is None or year_b is None:
        return None
    try:
        diff = abs(int(year_a) - int(year_b))
    except (TypeError, ValueError):
        return None
    return max(0.0, 1.0 - diff / 10.0)


def job_similarity(job_a: str | None, job_b: str | None) -> float | None:
    if not job_a or not job_b:
        return None
    return JaroWinkler.normalized_similarity(str(job_a), str(job_b))


def education_similarity(edu_a: str | None, edu_b: str | None) -> float | None:
    if not edu_a or not edu_b:
        return None
    return JaroWinkler.normalized_similarity(str(edu_a), str(edu_b))


def phone_exact(phone_a: str | None, phone_b: str | None) -> float | None:
    """Compares the +98-canonical form computed in ETL, so the +98/0 format
    noise doesn't break an otherwise decisive match."""
    if not phone_a or not phone_b:
        return None
    return 1.0 if phone_a == phone_b else 0.0


def email_exact(email_a: str | None, email_b: str | None) -> float | None:
    if not email_a or not email_b:
        return None
    return 1.0 if email_a == email_b else 0.0


def extract(a: dict, b: dict) -> dict:
    return {
        "city_sim": city_match(a.get("city"), b.get("city")),
        "birth_year_prox": birth_year_proximity(a.get("birth_year"), b.get("birth_year")),
        "job_sim": job_similarity(a.get("job_title"), b.get("job_title")),
        "education_sim": education_similarity(a.get("education"), b.get("education")),
        "phone_exact": phone_exact(a.get("phone_canonical"), b.get("phone_canonical")),
        "email_exact": email_exact(a.get("email_normalized"), b.get("email_normalized")),
    }
