"""
Phase 5 - dossier API.

Serves the end product of the pipeline: given one social account, return the
other accounts the system believes belong to the same person, the real
identity it maps to, and - critically - WHY, as a per-feature evidence
breakdown rather than a bare score.

Design notes:

  - Precomputed clusters and identity mappings are loaded from the parquet
    files scripts/train_mapper.py writes. Re-running blocking + scoring over
    14k accounts takes ~50s, which is fine as a batch job and unacceptable per
    request. Explanations, being cheap, are computed live.

  - Every confidence shown for a real-identity attribution is the top-1
    CALIBRATED probability, not the raw model score. On the deployed path
    (predicted clusters) 0.92 means right about 93% of the time; the raw score
    would have claimed 0.998 for the same cases.

  - The legacy single-platform /resolve endpoint is kept so the old VectorID
    flow still works.

The feedback endpoint stores operator confirm/reject decisions. Nothing
retrains automatically - the labels accumulate for a future active-learning
pass, and pretending otherwise would be worse than not having them.
"""
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import List, Optional

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.models.fusion_model import FellegiSunterFusion
from src.models.pair_features import AccountUniverse, extract_pair_features
from src.real_identity_mapper import (RealIdentityMapper, RegistryIndex,
                                      candidates_for_clusters)
from src.retriever import IdentityRetriever
from src.text_utils import latin_skeleton, normalizer, transliterate
from src.vector_db import QdrantManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback_labels.parquet")

app = FastAPI(title="IdentityGraph API", version="2.0.0",
              description="Cross-platform identity resolution and real-identity mapping")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# State, loaded once at startup
# ---------------------------------------------------------------------------
class ResolutionState:
    """Everything the dossier endpoints need, held in memory."""

    def __init__(self):
        self.universe = None
        self.index = None
        self.mapper = None
        self.pair_model = None
        self.cluster_of = {}      # record_id -> cluster_id
        self.members_of = {}      # cluster_id -> [record_id]
        self.identity_map = {}    # cluster_id -> [ranked registry matches]
        self.by_username = {}     # (platform, username) -> record_id
        self.loaded = False
        self.error = None

    def load(self):
        try:
            self.universe = AccountUniverse.load(DATA_DIR)
            self.index = RegistryIndex.load(DATA_DIR)

            model = FellegiSunterFusion.load(os.path.join(DATA_DIR, "mapper_model.json"))
            top1_cal = None
            cal_path = os.path.join(DATA_DIR, "mapper_top1_calibration.json")
            if os.path.exists(cal_path):
                with open(cal_path, encoding="utf-8") as f:
                    top1_cal = json.load(f)
            self.mapper = RealIdentityMapper(self.index, model, top1_calibration=top1_cal)
            self.pair_model = FellegiSunterFusion.load(os.path.join(DATA_DIR, "fusion_model.json"))

            clusters = pd.read_parquet(os.path.join(DATA_DIR, "resolution_clusters.parquet"))
            for row in clusters.itertuples(index=False):
                self.cluster_of[row.record_id] = int(row.cluster_id)
                self.members_of.setdefault(int(row.cluster_id), []).append(row.record_id)

            id_map = pd.read_parquet(os.path.join(DATA_DIR, "identity_map.parquet"))
            for row in id_map.sort_values("rank").itertuples(index=False):
                self.identity_map.setdefault(int(row.cluster_id), []).append({
                    "entity_id": row.entity_id,
                    "confidence": float(row.confidence),
                    "confidence_top1": (float(row.confidence_top1)
                                        if row.confidence_top1 is not None
                                        and not pd.isna(row.confidence_top1) else None),
                    # fraction of the total candidate mass this record holds;
                    # ~0.5 on the top two means the evidence identifies a PAIR
                    # of people, not one of them (the twin case)
                    "share": float(row.share) if not pd.isna(row.share) else None,
                    "full_name": row.full_name,
                    "city": row.city,
                    "birth_year": None if pd.isna(row.birth_year) else int(row.birth_year),
                    "job": row.job,
                    "education": row.education,
                    "rank": int(row.rank),
                })

            for rid, acct in self.universe.accounts.items():
                self.by_username[(acct["platform"], str(acct["username"]).lower())] = rid

            self.loaded = True
            logger.info("Loaded %d accounts, %d clusters, %d registry records",
                        len(self.universe.accounts), len(self.members_of), len(self.index))
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            logger.error("Failed to load resolution state: %s", e)


state = ResolutionState()


@app.on_event("startup")
def _startup():
    state.load()


def _require_loaded():
    if not state.loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Resolution data not loaded ({state.error}). "
                   f"Run: python scripts/train_mapper.py",
        )


def _clean(value):
    """Parquet/numpy scalars -> JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _account_summary(record_id: str) -> dict:
    acct = state.universe.accounts[record_id]
    posts = acct.get("posts")
    post_list = list(posts) if posts is not None else []
    return {
        "record_id": record_id,
        "platform": acct["platform"],
        "username": _clean(acct.get("username")),
        "display_name": _clean(acct.get("display_name")),
        "bio": _clean(acct.get("bio")),
        "city": _clean(acct.get("city")),
        "birth_year": _clean(acct.get("birth_year")),
        "job_title": _clean(acct.get("job_title")),
        "education": _clean(acct.get("education")),
        "phone_number": _clean(acct.get("phone_number")),
        "email": _clean(acct.get("email")),
        "num_posts": _clean(acct.get("num_posts")),
        "graph_degree": _clean(acct.get("graph_degree")),
        "posts": [{"text": p.get("text"), "timestamp": p.get("timestamp")} for p in post_list[:5]],
    }


# ---------------------------------------------------------------------------
# Dossier
# ---------------------------------------------------------------------------
# NOTE: this must be declared BEFORE /dossier/{platform}/{username}. FastAPI
# matches routes in declaration order, so the generic two-segment route would
# otherwise swallow "/dossier/by-record/<id>" as platform="by-record".
@app.get("/dossier/by-record/{record_id}", summary="Full identity dossier by record id")
def get_dossier_by_record(record_id: str):
    _require_loaded()
    if record_id not in state.universe.accounts:
        raise HTTPException(status_code=404, detail=f"Unknown record_id {record_id}")
    return build_dossier(record_id)


@app.get("/dossier/{platform}/{username}", summary="Full identity dossier for one account")
def get_dossier(platform: str, username: str):
    _require_loaded()
    record_id = state.by_username.get((platform.lower(), username.lower()))
    if not record_id:
        raise HTTPException(status_code=404, detail=f"No {platform} account '{username}'")
    return build_dossier(record_id)


def build_dossier(record_id: str) -> dict:
    cluster_id = state.cluster_of.get(record_id)
    members = state.members_of.get(cluster_id, [record_id])

    # Why each linked account was linked: the same per-feature log-odds
    # breakdown the fusion model used to accept the pair.
    linked = []
    for other in members:
        if other == record_id:
            continue
        features = extract_pair_features(record_id, other, state.universe)
        score = float(state.pair_model.predict_proba([features])[0])
        linked.append({
            "account": _account_summary(other),
            "match_confidence": score,
            "evidence": [
                {"feature": name, "level": level, "weight_bits": round(weight, 3)}
                for name, level, weight in state.pair_model.explain(features)
                if abs(weight) > 1e-9
            ],
        })
    linked.sort(key=lambda x: -x["match_confidence"])

    candidates = state.identity_map.get(cluster_id, [])
    real_identity = None
    if candidates:
        top = candidates[0]
        members_data = [state.universe.accounts[m] for m in members]
        ranked = state.mapper.rank(members_data, [c["entity_id"] for c in candidates], top_n=len(candidates))
        evidence = ranked[0]["evidence"] if ranked else []
        runner_up = candidates[1] if len(candidates) > 1 else None
        real_identity = {
            **top,
            "evidence": [
                {"feature": name, "level": level, "weight_bits": round(weight, 3)}
                for name, level, weight in evidence if abs(weight) > 1e-9
            ],
            # An operator needs to see a near-tie as a near-tie. The evidence
            # panel alone can't show it: a twin pair produces an identical,
            # entirely convincing list of reasons for both people.
            "contested": bool(runner_up and runner_up["confidence"] > 0.9 * top["confidence"]),
            "alternatives": candidates[1:],
        }

    return {
        "query_account": _account_summary(record_id),
        "cluster_id": cluster_id,
        "linked_accounts": linked,
        "platforms_covered": sorted({state.universe.accounts[m]["platform"] for m in members}),
        "real_identity": real_identity,
    }


@app.get("/search", summary="Find accounts by name or username substring")
def search(q: str, limit: int = 20):
    _require_loaded()
    needle = q.strip().lower()
    if not needle:
        return {"results": []}

    # A Latin query has to survive the same problem the matcher does: the
    # operator types "fatemeh", the account calls itself "fatmah", and Persian
    # script never specified which of them was right. So a literal miss falls
    # back to the consonant skeleton, and a Latin query is also compared
    # against the transliterated display name - an operator searching for a
    # person by their romanised name should not have to know which platform
    # wrote it in Persian.
    skeleton = latin_skeleton(needle)
    use_skeleton = len(skeleton) >= 3

    results = []
    for rid, acct in state.universe.accounts.items():
        name = str(acct.get("display_name") or "").lower()
        user = str(acct.get("username") or "").lower()
        hit = needle in name or needle in user
        if not hit and use_skeleton:
            hit = (skeleton in latin_skeleton(user)
                   or skeleton in latin_skeleton(transliterate(name)))
        if hit:
            results.append({
                "record_id": rid,
                "platform": acct["platform"],
                "username": _clean(acct.get("username")),
                "display_name": _clean(acct.get("display_name")),
                "cluster_size": len(state.members_of.get(state.cluster_of.get(rid), [rid])),
            })
            if len(results) >= limit:
                break
    return {"results": results}


@app.get("/stats", summary="Corpus and resolution statistics")
def stats():
    _require_loaded()
    sizes = [len(m) for m in state.members_of.values()]
    platforms = {}
    for acct in state.universe.accounts.values():
        platforms[acct["platform"]] = platforms.get(acct["platform"], 0) + 1
    return {
        "accounts": len(state.universe.accounts),
        "platforms": platforms,
        "clusters": len(state.members_of),
        "multi_account_clusters": sum(1 for s in sizes if s > 1),
        "largest_cluster": max(sizes) if sizes else 0,
        "registry_records": len(state.index),
        "feedback_labels": _feedback_count(),
    }


# ---------------------------------------------------------------------------
# Operator feedback (active-learning substrate)
# ---------------------------------------------------------------------------
class Feedback(BaseModel):
    record_id: str
    decision: str  # "confirm" | "reject"
    target_type: str  # "linked_account" | "real_identity"
    target_id: str  # the other record_id, or the registry entity_id
    note: Optional[str] = None


@app.post("/feedback", summary="Record an operator's confirm/reject decision")
def submit_feedback(feedback: Feedback):
    if feedback.decision not in ("confirm", "reject"):
        raise HTTPException(status_code=400, detail="decision must be 'confirm' or 'reject'")
    if feedback.target_type not in ("linked_account", "real_identity"):
        raise HTTPException(status_code=400, detail="target_type must be 'linked_account' or 'real_identity'")

    row = {
        "feedback_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_id": feedback.record_id,
        "decision": feedback.decision,
        "target_type": feedback.target_type,
        "target_id": feedback.target_id,
        "note": feedback.note,
    }
    existing = pd.read_parquet(FEEDBACK_PATH) if os.path.exists(FEEDBACK_PATH) else pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_parquet(FEEDBACK_PATH, index=False)
    return {"status": "recorded", "total_labels": len(updated)}


@app.get("/feedback", summary="All recorded feedback")
def list_feedback(limit: int = 100):
    if not os.path.exists(FEEDBACK_PATH):
        return {"labels": [], "total": 0}
    df = pd.read_parquet(FEEDBACK_PATH)
    return {"labels": df.tail(limit).to_dict("records"), "total": len(df)}


def _feedback_count() -> int:
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    return len(pd.read_parquet(FEEDBACK_PATH))


# ---------------------------------------------------------------------------
# Legacy single-platform endpoints (the original VectorID flow)
# ---------------------------------------------------------------------------
class ProfileInput(BaseModel):
    full_name: str
    username: Optional[str] = ""
    email: Optional[str] = ""
    phone_number: Optional[str] = ""
    city: Optional[str] = ""
    bio: Optional[str] = ""
    birth_year: Optional[int] = None


class MatchResult(BaseModel):
    record_id: str
    score: float
    profile: dict


_legacy_qdrant = QdrantManager()
_legacy_retriever = IdentityRetriever(_legacy_qdrant)


@app.post("/resolve", response_model=List[MatchResult],
          summary="[legacy] Single-platform duplicate resolution")
def resolve_identity(profile: ProfileInput):
    payload = profile.model_dump()
    name_text = normalizer.normalize(profile.full_name) if profile.full_name else ""
    city_text = normalizer.normalize(profile.city) if profile.city else ""
    bio_text = normalizer.normalize(profile.bio) if profile.bio else ""
    payload["full_name"] = normalizer.normalize(payload.get("full_name", ""))
    payload["city"] = normalizer.normalize(payload.get("city", ""))
    payload["bio"] = normalizer.normalize(payload.get("bio", ""))
    try:
        return _legacy_retriever.resolve_profile(payload, name_text, city_text, bio_text)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to resolve: %s", e)
        raise HTTPException(status_code=500, detail="Resolution failed.")
