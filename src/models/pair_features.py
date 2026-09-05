"""
Phase 3 - assembles every modality's features for a candidate pair.

Loading the account universe once (accounts, style profiles, hour profiles,
vectors) and passing it around avoids recomputing per-account state for every
pair - each account participates in many candidate pairs.
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import graph_model, metadata_model, name_model, stylometry_model, temporal_model, text_model

PLATFORMS = ["twitter", "instagram", "telegram", "linkedin"]


class AccountUniverse:
    """All accounts plus the per-account state the pair features need."""

    def __init__(self, accounts: dict, style_profiles: dict, hour_profiles: dict, vectors: dict):
        self.accounts = accounts
        self.style_profiles = style_profiles
        self.hour_profiles = hour_profiles
        self.vectors = vectors

    @classmethod
    def load(cls, data_dir: str, vectors: dict | None = None) -> "AccountUniverse":
        accounts = {}
        for platform in PLATFORMS:
            cleaned = pd.read_parquet(os.path.join(data_dir, "cleaned", f"{platform}.parquet"))
            feats = pd.read_parquet(os.path.join(data_dir, "graph_features", f"{platform}.parquet"))
            merged = cleaned.merge(feats, on="record_id", how="left")
            for row in merged.to_dict("records"):
                accounts[row["record_id"]] = row

        style_profiles = {rid: stylometry_model.style_profile(acct) for rid, acct in accounts.items()}
        hour_profiles = {rid: temporal_model.hour_profile(acct) for rid, acct in accounts.items()}
        return cls(accounts, style_profiles, hour_profiles, vectors or {})

    def __len__(self):
        return len(self.accounts)


def extract_pair_features(record_id_a: str, record_id_b: str, universe: AccountUniverse) -> dict:
    a = universe.accounts[record_id_a]
    b = universe.accounts[record_id_b]

    features = {}
    features.update(name_model.extract(a, b))
    features.update(metadata_model.extract(a, b))
    features.update(text_model.extract(a, b, universe.vectors))
    features.update(stylometry_model.extract(a, b, universe.style_profiles))
    features.update(temporal_model.extract(a, b, universe.hour_profiles))
    features.update(graph_model.extract(a, b))
    return features


# Which features belong to which modality - used by the ablation study to
# measure what each modality actually contributes.
MODALITY_FEATURES = {
    "name": ["name_jw", "name_token_sort", "name_phonetic", "name_nickname",
             "username_jw", "username_nodigits", "username_vs_name"],
    "metadata": ["city_sim", "birth_year_prox", "job_sim", "education_sim",
                 "phone_exact", "email_exact"],
    "text": ["bio_cosine", "posts_cosine"],
    "stylometry": ["style_punctuation", "style_emoji", "style_finglish", "style_function_words"],
    "temporal": ["temporal_peak_match", "temporal_histogram_cosine"],
    "graph": ["graph_degree_sim", "graph_clustering_sim", "graph_neighbor_degree_sim"],
}
