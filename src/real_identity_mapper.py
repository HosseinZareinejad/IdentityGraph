"""
Phase 5 - mapping a virtual identity (a cluster of platform accounts) to a
real identity in the secondary registry.

This is the step the whole proposal was actually about. Phases 1-4 answer
"which accounts belong to the same person"; this answers "and who is that
person", against a registry that stands in for an official database.

Two things make it structurally different from cross-platform pair matching:

  1. THE EVIDENCE IS POOLED. A cluster knows collectively what no single
     account knows: the LinkedIn account supplies job and education, Telegram
     supplies a phone, Instagram supplies a city. Each feature is therefore
     computed as the best available comparison across all cluster members
     rather than from one record.

  2. THE HARD CASE IS BUILT IN. Phase 1 planted 200 "twins" - distinct people
     sharing full name, city and birth year. Against a registry, name + city +
     birth year cannot separate them by construction; only job, education,
     phone or email can. Those live almost exclusively on LinkedIn and
     Telegram, so a cluster containing neither is genuinely undecidable, not
     merely difficult. Reporting twin accuracy separately from overall
     accuracy is the honest way to show this, and it is the sharpest available
     answer to the feasibility question the client actually asked.

The scorer is the same FellegiSunterFusion used in Phase 3 (missing evidence
stays neutral, weights are learned, the output decomposes into per-feature
log-odds), trained on its own feature set with its own weights.
"""
import os

import numpy as np
import pandas as pd
from rapidfuzz import process as rf_process
from rapidfuzz.distance import JaroWinkler

from src.models import metadata_model, name_model
from src.text_utils import canonicalize_phone, compute_phonetic_key, transliterate

# Features reuse Phase 3's names so FEATURE_BINS in fusion_model.py applies
# unchanged - the comparisons mean the same thing, only the operands differ
# (cluster-vs-registry instead of account-vs-account).
MAPPER_FEATURES = [
    "name_jw", "name_token_sort", "name_phonetic", "name_nickname",
    "username_vs_name", "city_sim", "birth_year_prox", "job_sim",
    "education_sim", "phone_exact", "email_exact",
]

MAX_PHONETIC_BLOCK = 60


class RegistryIndex:
    """The secondary registry plus the derived keys needed to block against it."""

    def __init__(self, records: dict):
        self.records = records
        self.entity_ids = list(records.keys())
        self.names = [records[e]["full_name"] or "" for e in self.entity_ids]

        self.by_phonetic = {}
        self.by_phone = {}
        self.by_email = {}
        for entity_id, rec in records.items():
            self.by_phonetic.setdefault(rec["phonetic_key"], []).append(entity_id)
            if rec["phone_canonical"]:
                self.by_phone.setdefault(rec["phone_canonical"], []).append(entity_id)
            if rec["email_normalized"]:
                self.by_email.setdefault(rec["email_normalized"], []).append(entity_id)

    @classmethod
    def load(cls, data_dir: str) -> "RegistryIndex":
        df = pd.read_parquet(os.path.join(data_dir, "real_identity_registry.parquet"))
        records = {}
        for row in df.to_dict("records"):
            full_name = row.get("full_name") or ""
            records[row["entity_id"]] = {
                "entity_id": row["entity_id"],
                "full_name": full_name,
                "first_name": row.get("first_name"),
                "last_name": row.get("last_name"),
                "birth_year": row.get("birth_year"),
                "city": row.get("city"),
                "job": row.get("job"),
                "education": row.get("education"),
                "twin_of": row.get("twin_of"),
                "phonetic_key": compute_phonetic_key(full_name),
                "name_translit": transliterate(name_model.strip_titles(full_name)),
                "phone_canonical": canonicalize_phone(row.get("phone_number")),
                "email_normalized": (row.get("email") or "").strip().lower() or None,
            }
        return cls(records)

    def __len__(self):
        return len(self.records)


def extract_cluster_features(members: list[dict], registry_rec: dict) -> dict:
    """Compare a whole cluster against one registry record.

    Every feature is the BEST evidence any member can offer. `None` means no
    member had the field at all, which the fusion model scores as neutral
    rather than as a mismatch.
    """
    def best(values):
        present = [v for v in values if v is not None]
        return max(present) if present else None

    reg_name = registry_rec["full_name"]
    reg_key = registry_rec["phonetic_key"]

    features = {
        "name_jw": best([name_model.name_similarity(m.get("display_name"), reg_name) for m in members]),
        "name_token_sort": best([name_model.name_token_sort_similarity(m.get("display_name"), reg_name)
                                 for m in members]),
        "name_phonetic": best([name_model.phonetic_key_match(m.get("name_phonetic_key"), reg_key)
                               for m in members]),
        "name_nickname": best([name_model.nickname_compatible(m.get("display_name"), reg_name)
                               for m in members]),
        # the registry always holds the legal name, so the direction is fixed:
        # member handle vs registry name
        "username_vs_name": best([name_model.username_vs_name(m.get("username"), reg_name)
                                  for m in members]),
        "city_sim": best([metadata_model.city_match(m.get("city"), registry_rec["city"]) for m in members]),
        "birth_year_prox": best([metadata_model.birth_year_proximity(m.get("birth_year"),
                                                                    registry_rec["birth_year"])
                                 for m in members]),
        "job_sim": best([metadata_model.job_similarity(m.get("job_title"), registry_rec["job"])
                         for m in members]),
        "education_sim": best([metadata_model.education_similarity(m.get("education"),
                                                                   registry_rec["education"])
                               for m in members]),
        "phone_exact": best([metadata_model.phone_exact(m.get("phone_canonical"),
                                                        registry_rec["phone_canonical"])
                             for m in members]),
        "email_exact": best([metadata_model.email_exact(m.get("email_normalized"),
                                                        registry_rec["email_normalized"])
                             for m in members]),
    }
    return features


def candidates_for_clusters(clusters: list[set], universe, index: RegistryIndex,
                            top_k: int = 10, chunk_size: int = 2000) -> list[list[str]]:
    """Registry candidates per cluster, unioned over three blocking paths.

    Fuzzy name matching is batched into one cdist per chunk rather than a call
    per cluster - 14k account names against a 5k registry is 70M comparisons,
    which rapidfuzz's C backend handles in one pass but not in 14k separate
    Python-level calls.
    """
    member_names, member_owner = [], []
    for cluster_idx, cluster in enumerate(clusters):
        for rid in cluster:
            acct = universe.accounts[rid]
            member_names.append(name_model.strip_titles(str(acct.get("display_name") or "")))
            member_owner.append(cluster_idx)

    per_cluster = [set() for _ in clusters]

    for start in range(0, len(member_names), chunk_size):
        chunk = member_names[start:start + chunk_size]
        sim = rf_process.cdist(chunk, index.names, scorer=JaroWinkler.normalized_similarity,
                               workers=-1, dtype=np.float32)
        k = min(top_k, sim.shape[1])
        top = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        for row_idx, cols in enumerate(top):
            cluster_idx = member_owner[start + row_idx]
            for col in cols:
                per_cluster[cluster_idx].add(index.entity_ids[col])

    # exact-key paths: phonetic name, phone, email
    for cluster_idx, cluster in enumerate(clusters):
        for rid in cluster:
            acct = universe.accounts[rid]
            key = acct.get("name_phonetic_key")
            if key:
                block = index.by_phonetic.get(key, [])
                if len(block) <= MAX_PHONETIC_BLOCK:
                    per_cluster[cluster_idx].update(block)
            phone = acct.get("phone_canonical")
            if phone:
                per_cluster[cluster_idx].update(index.by_phone.get(phone, []))
            email = acct.get("email_normalized")
            if email:
                per_cluster[cluster_idx].update(index.by_email.get(email, []))

    return [sorted(c) for c in per_cluster]


class RealIdentityMapper:
    """Ranks registry records for a cluster using a trained fusion model.

    Two different probabilities come out of this, and conflating them is a
    real trap:

      confidence          - P(this candidate is the right one), calibrated
                            over ALL candidates. Correct for comparing
                            candidates within one ranking.
      confidence_top1     - P(the top-ranked candidate is right), which is a
                            different and strictly easier question, because
                            the top candidate is an argmax rather than a
                            random draw. Measured on held-out clusters, the
                            raw score badly understates this: rows scoring
                            0.18 were right 81% of the time.

    The dossier UI shows an attribution to a real person, so it must display
    confidence_top1. The separate isotonic fit below is what makes that number
    mean what it says.
    """

    def __init__(self, index: RegistryIndex, model, top1_calibration: dict | None = None):
        self.index = index
        self.model = model
        self.top1_calibration = top1_calibration

    def calibrate_top1(self, raw_confidence: float, share: float) -> float:
        """Map (top score, share of the candidate mass) to P(top-1 correct).

        `share` is p1 / sum(p) over the candidate list: the posterior on the
        top candidate under the assumption that exactly one candidate in the
        list is the right person. It is what encodes AMBIGUITY, and it is the
        thing a score-only calibrator is blind to.

        This matters exactly where the system is weakest. Phase 1 planted
        twins - distinct people sharing name, city and birth year - and a
        cluster with no job, education, phone or email cannot separate them
        even in principle. Both twins then score high and near-identically,
        the argmax picks one at random, and a score-only calibrator happily
        reports 93% because the winning score is high. The share for that
        cluster is ~0.5, which says precisely what is true: the evidence
        supports the pair, not the individual.
        """
        if not self.top1_calibration:
            return raw_confidence
        cal = self.top1_calibration
        if cal.get("kind") == "logistic+isotonic":
            z = (cal["intercept"]
                 + cal["coef_score"] * float(_logit(raw_confidence))
                 + cal["coef_share"] * float(_logit(share)))
            combined = 1.0 / (1.0 + np.exp(-z))
            return float(np.interp(combined, cal["x"], cal["y"]))
        return float(np.interp(raw_confidence, cal["x"], cal["y"]))

    def rank(self, members: list[dict], candidate_entity_ids: list[str], top_n: int = 5) -> list[dict]:
        if not candidate_entity_ids:
            return []
        rows = [extract_cluster_features(members, self.index.records[e]) for e in candidate_entity_ids]
        probabilities = self.model.predict_proba(rows)
        total = float(probabilities.sum())
        order = np.argsort(-probabilities)[:top_n]

        top_score = float(probabilities[order[0]])
        share = top_score / total if total > 0 else 1.0

        results = []
        for rank, i in enumerate(order):
            results.append({
                "entity_id": candidate_entity_ids[i],
                "confidence": float(probabilities[i]),
                # only the top-ranked entry gets the top-1 calibration; for the
                # rest the question it answers doesn't apply
                "confidence_top1": self.calibrate_top1(top_score, share) if rank == 0 else None,
                # how much of the total candidate mass this entry holds - the
                # UI shows it so an operator can see a two-way tie for what it is
                "share": float(probabilities[i]) / total if total > 0 else 1.0,
                "registry": self.index.records[candidate_entity_ids[i]],
                "evidence": self.model.explain(rows[i]),
                "features": rows[i],
            })
        return results


_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_top1_calibration(top1_confidences: list[float], shares: list[float],
                         correct: list[int]) -> dict:
    """Fit P(top-1 correct) from the top score AND the candidate-mass share.

    Two inputs rather than one, because they answer different questions: the
    score says "is this person a good fit for the evidence", the share says
    "is anyone else an equally good fit". A twin cluster scores high on the
    first and ~0.5 on the second, and only the second is telling the truth.

    Two stages, because neither alone does the job. Isotonic regression
    calibrates beautifully but is one-dimensional, so it cannot see the share
    at all. A two-input logistic sees both but imposes a parametric shape that
    fits this data badly - tried on its own it reported 0.33 on a bin that was
    right 1% of the time. So the logistic is used only to COMBINE the two
    inputs into one ranking statistic, and isotonic then calibrates that
    statistic non-parametrically.

    The two stages are fitted on disjoint halves of the calibration data. An
    isotonic curve fitted to the same rows the logistic just fitted would
    reproduce them almost exactly and report a calibration that doesn't hold.

    MUST be fitted on clusters the ranking model didn't train on, and on
    PREDICTED clusters rather than ground-truth ones - the deployed system
    never sees a perfect cluster.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    x = np.column_stack([_logit(top1_confidences), _logit(shares)])
    y = np.asarray(correct, dtype=int)
    identity = {"kind": "logistic+isotonic", "intercept": 0.0, "coef_score": 1.0,
                "coef_share": 0.0, "x": [0.0, 1.0], "y": [0.0, 1.0]}
    if len(np.unique(y)) < 2 or len(y) < 100:
        return identity

    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    half = len(order) // 2
    fit_idx, cal_idx = order[:half], order[half:]
    if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[cal_idx])) < 2:
        return identity

    logistic = LogisticRegression(max_iter=1000).fit(x[fit_idx], y[fit_idx])
    combined = logistic.predict_proba(x[cal_idx])[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(combined, y[cal_idx].astype(float))
    grid = np.linspace(0.0, 1.0, 201)

    return {
        "kind": "logistic+isotonic",
        "intercept": float(logistic.intercept_[0]),
        "coef_score": float(logistic.coef_[0][0]),
        "coef_share": float(logistic.coef_[0][1]),
        "x": grid.tolist(),
        "y": iso.predict(grid).tolist(),
        "n_logistic": int(len(fit_idx)),
        "n_isotonic": int(len(cal_idx)),
    }
