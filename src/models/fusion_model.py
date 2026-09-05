"""
Phase 3 - Fellegi-Sunter style fusion scorer.

Replaces the hand-tuned linear formula in src/scorer.py
(0.4*vector + 0.4*name + 0.2*username, with missing fields scored 0.0).

Three properties that formula lacked:

  1. MISSING IS NEUTRAL. Each feature is bucketed into comparison levels, one
     of which is "missing". The missing level carries a match weight of
     exactly 0 - it moves the score neither up nor down. The old scorer gave
     a missing field a similarity of 0.0, i.e. it actively argued AGAINST a
     match whenever data was absent. Given Phase 2's coverage matrix (job and
     education only exist on LinkedIn, phone almost only on Telegram, birth
     year almost nowhere) that bug penalised precisely the sparse
     cross-platform pairs the system exists to find.

  2. WEIGHTS ARE LEARNED, NOT GUESSED. For each comparison level the model
     estimates
         m = P(level | the pair IS the same person)
         u = P(level | the pair is NOT the same person)
     and uses the classic Fellegi-Sunter match weight log2(m/u). A level that
     is common among matches and rare among non-matches earns a large positive
     weight; one that is equally common in both earns ~0. Laplace smoothing
     keeps a level that never appears in one class from producing an infinite
     weight.

  3. THE OUTPUT IS AN INTERPRETABLE, CALIBRATED PROBABILITY. The total score
     is the prior log-odds plus the sum of per-level weights, so any decision
     can be explained as "+4.1 bits from an exact phonetic name match, -1.2
     bits from conflicting birth years, 0 from the missing phone". Converting
     the log-odds back through a sigmoid gives a probability that can be
     checked against observed match rates (see calibration_report).

This is Fellegi-Sunter with supervised weights rather than EM-estimated ones
(we have ground-truth labels, so EM is unnecessary), which also lets it cover
heterogeneous features - graph, temporal and stylometric similarities - that
do not fit a column-comparison DSL like splink's.
"""
import json
import math

import numpy as np
from sklearn.isotonic import IsotonicRegression

# Comparison-level bucket edges per feature. Values are upper bounds; a value
# <= edge falls in that bucket. Chosen to be coarse - fine-grained bins would
# overfit and make the weight table unreadable.
DEFAULT_BINS = [0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
BINARY_BINS = [0.5, 1.01]

FEATURE_BINS = {
    "name_jw": DEFAULT_BINS,
    "name_token_sort": DEFAULT_BINS,
    "name_phonetic": BINARY_BINS,
    "name_nickname": BINARY_BINS,
    "username_jw": DEFAULT_BINS,
    "username_nodigits": DEFAULT_BINS,
    "username_vs_name": DEFAULT_BINS,
    "city_sim": DEFAULT_BINS,
    "birth_year_prox": [0.5, 0.8, 0.95, 1.01],
    "job_sim": DEFAULT_BINS,
    "education_sim": DEFAULT_BINS,
    "phone_exact": BINARY_BINS,
    "email_exact": BINARY_BINS,
    "bio_cosine": [0.3, 0.5, 0.7, 0.85, 1.01],
    "posts_cosine": [0.3, 0.5, 0.7, 0.85, 1.01],
    "style_punctuation": [0.5, 0.8, 0.95, 1.01],
    "style_emoji": [0.5, 0.8, 0.95, 1.01],
    "style_finglish": [0.5, 0.8, 0.95, 1.01],
    "style_function_words": [0.3, 0.6, 0.8, 1.01],
    "temporal_peak_match": [0.5, 0.7, 0.85, 0.95, 1.01],
    "temporal_histogram_cosine": [0.3, 0.5, 0.7, 0.85, 1.01],
    "graph_degree_sim": [0.4, 0.7, 0.9, 1.01],
    "graph_clustering_sim": [0.4, 0.7, 0.9, 1.01],
    "graph_neighbor_degree_sim": [0.4, 0.7, 0.9, 1.01],
}

MISSING_LEVEL = "missing"


def bucket(feature: str, value) -> str:
    """Map a raw feature value to a comparison level label."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING_LEVEL
    edges = FEATURE_BINS.get(feature, DEFAULT_BINS)
    for edge in edges:
        if value <= edge:
            return f"<={edge}"
    return f">{edges[-1]}"


class FellegiSunterFusion:
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.weights: dict[str, dict[str, float]] = {}
        self.prior_log_odds: float = 0.0
        self.features: list[str] = []
        # Isotonic calibration curve, stored as interpolation points so the
        # model round-trips through JSON without pickling sklearn objects.
        self.calibration: dict[str, list[float]] | None = None

    def fit(self, feature_rows: list[dict], labels: list[int]) -> "FellegiSunterFusion":
        labels = np.asarray(labels)
        n_match = int(labels.sum())
        n_nonmatch = int(len(labels) - n_match)
        if n_match == 0 or n_nonmatch == 0:
            raise ValueError("fit() needs both matching and non-matching pairs")

        self.prior_log_odds = math.log2(n_match / n_nonmatch)
        self.features = sorted({f for row in feature_rows for f in row})

        for feature in self.features:
            match_counts: dict[str, int] = {}
            nonmatch_counts: dict[str, int] = {}
            for row, label in zip(feature_rows, labels):
                level = bucket(feature, row.get(feature))
                target = match_counts if label else nonmatch_counts
                target[level] = target.get(level, 0) + 1

            levels = set(match_counts) | set(nonmatch_counts)
            n_levels = max(len(levels), 1)
            feature_weights = {}
            for level in levels:
                # Laplace-smoothed m and u probabilities
                m = (match_counts.get(level, 0) + self.smoothing) / (n_match + self.smoothing * n_levels)
                u = (nonmatch_counts.get(level, 0) + self.smoothing) / (n_nonmatch + self.smoothing * n_levels)
                feature_weights[level] = math.log2(m / u)
            # By definition, absence of evidence contributes nothing.
            feature_weights[MISSING_LEVEL] = 0.0
            self.weights[feature] = feature_weights
        return self

    def explain(self, row: dict) -> list[tuple[str, str, float]]:
        """Per-feature evidence contributions, largest magnitude first - this
        is what the UI's explainability panel renders."""
        contributions = []
        for feature in self.features:
            level = bucket(feature, row.get(feature))
            weight = self.weights.get(feature, {}).get(level, 0.0)
            if weight != 0.0:
                contributions.append((feature, level, weight))
        contributions.sort(key=lambda t: abs(t[2]), reverse=True)
        return contributions

    def score_log_odds(self, row: dict) -> float:
        total = self.prior_log_odds
        for feature in self.features:
            level = bucket(feature, row.get(feature))
            total += self.weights.get(feature, {}).get(level, 0.0)
        return total

    def fit_calibration(self, rows: list[dict], labels: list[int]) -> "FellegiSunterFusion":
        """Learn a monotonic map from raw score to observed match frequency.

        Fellegi-Sunter assumes the comparison features are conditionally
        independent given match status. Ours are emphatically not - name_jw,
        name_token_sort, name_phonetic and the username features all restate
        "the names look alike" - so summing their log-odds counts the same
        evidence several times and drives scores to saturation. Uncalibrated,
        the model claimed 99.7% on a bucket that was really 83.5% correct.

        Isotonic regression fixes the mapping without touching the weights, so
        the per-feature explanation stays intact while the number a human
        reads becomes trustworthy. MUST be fitted on data disjoint from the
        weight-fitting set, or it just relearns the training noise.
        """
        raw = np.array([self.score_log_odds(r) for r in rows], dtype=np.float64)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw, np.asarray(labels, dtype=np.float64))
        self.calibration = {
            "x": [float(v) for v in iso.X_thresholds_],
            "y": [float(v) for v in iso.y_thresholds_],
        }
        return self

    def predict_proba(self, rows: list[dict]) -> np.ndarray:
        log_odds = np.array([self.score_log_odds(r) for r in rows], dtype=np.float64)
        if self.calibration:
            return np.interp(log_odds, self.calibration["x"], self.calibration["y"])
        return 1.0 / (1.0 + np.exp(-log_odds * math.log(2)))  # log2 odds -> probability

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "smoothing": self.smoothing,
                "prior_log_odds": self.prior_log_odds,
                "features": self.features,
                "weights": self.weights,
                "calibration": self.calibration,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "FellegiSunterFusion":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = cls(smoothing=data["smoothing"])
        model.prior_log_odds = data["prior_log_odds"]
        model.features = data["features"]
        model.weights = data["weights"]
        model.calibration = data.get("calibration")
        return model


def brier_score(probabilities: np.ndarray, labels) -> float:
    """Mean squared error of the probabilities - lower is better. Sensitive to
    exactly the overconfidence that calibration is meant to remove."""
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probabilities - labels) ** 2))


def calibration_report(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Do the predicted probabilities mean what they say? Compares predicted
    probability against the observed match rate in each bin."""
    labels = np.asarray(labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    report = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probabilities >= lo) & (probabilities < hi if i < n_bins - 1 else probabilities <= hi)
        if not mask.any():
            continue
        report.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "predicted": float(probabilities[mask].mean()),
            "observed": float(labels[mask].mean()),
        })
    return report
