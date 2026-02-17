from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uuid
import sys
import os

# Fix Proxy issue on Windows
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# Ensure the parent directory is in the sys path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_db import QdrantManager
from src.retriever import IdentityRetriever
from src.text_utils import normalizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VectorID API", description="Identity Resolution System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant_manager = QdrantManager()
retriever = IdentityRetriever(qdrant_manager)

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

@app.post("/ingest", summary="Ingest a new profile into the system")
def ingest_profile(profile: ProfileInput):
    record_id = str(uuid.uuid4())
    payload = profile.model_dump()
    payload['entity_id'] = record_id # Initially, assume it's a new entity
    
    name_text = normalizer.normalize(profile.full_name) if profile.full_name else "نامشخص"
    city_text = normalizer.normalize(profile.city) if profile.city else "نامشخص"
    bio_text = normalizer.normalize(profile.bio) if profile.bio else "بدون بیوگرافی"
    
    try:
        qdrant_manager.insert_profile(record_id, payload, name_text, city_text, bio_text)
        return {"status": "success", "record_id": record_id, "message": "Profile indexed successfully."}
    except Exception as e:
        logger.error(f"Failed to ingest: {e}")
        raise HTTPException(status_code=500, detail="Database insertion failed.")

@app.post("/resolve", response_model=List[MatchResult], summary="Resolve identity against existing profiles")
def resolve_identity(profile: ProfileInput):
    payload = profile.model_dump()
    # Normalize input fields for query
    name_text = normalizer.normalize(profile.full_name) if profile.full_name else ""
    city_text = normalizer.normalize(profile.city) if profile.city else ""
    bio_text = normalizer.normalize(profile.bio) if profile.bio else ""
    
    # Also normalize the payload fields so EnsembleScorer gets the clean versions
    payload['full_name'] = normalizer.normalize(payload.get('full_name', ''))
    payload['city'] = normalizer.normalize(payload.get('city', ''))
    payload['bio'] = normalizer.normalize(payload.get('bio', ''))
    
    try:
        matches = retriever.resolve_profile(payload, name_text, city_text, bio_text)
        return matches
    except Exception as e:
        logger.error(f"Failed to resolve: {e}")
        raise HTTPException(status_code=500, detail="Resolution failed.")
