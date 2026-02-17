from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config import settings
from src.scorer import EnsembleScorer
import logging

logger = logging.getLogger(__name__)

class IdentityRetriever:
    def __init__(self, qdrant_manager):
        self.qdrant = qdrant_manager

    def resolve_profile(self, payload: dict, name_text: str, city_text: str, bio_text: str, limit: int = 50) -> list[dict]:
        """
        Finds matching profiles for a given new profile data.
        Returns a sorted list of matches with scores.
        """
        candidates = {}
        
        # 1. Semantic Search (Multi-Vector)
        vectors = []
        if name_text and name_text.strip():
            vectors.append(("name", self.qdrant.embed_text(name_text)))
        if city_text and city_text.strip():
            vectors.append(("city", self.qdrant.embed_text(city_text)))
        if bio_text and bio_text.strip():
            vectors.append(("bio", self.qdrant.embed_text(bio_text)))
            
        import httpx
        for vec_name, vec in vectors:
            try:
                res = httpx.post(
                    f"http://127.0.0.1:6333/collections/{self.qdrant.collection_name}/points/search",
                    json={
                        "vector": {"name": vec_name, "vector": vec},
                        "limit": limit,
                        "with_payload": True,
                        "with_vector": False
                    },
                    timeout=10.0
                )
                res.raise_for_status()
                semantic_hits = res.json().get("result", [])
                
                for hit in semantic_hits:
                    cand_id = str(hit["id"])
                    score = hit["score"]
                    if cand_id not in candidates:
                        candidates[cand_id] = {"payload": hit["payload"], "vector_score": score}
                    else:
                        candidates[cand_id]["vector_score"] = max(candidates[cand_id]["vector_score"], score)
            except Exception as e:
                logger.error(f"Semantic search failed for {vec_name}: {e}")
            
        # 2. Exact Match Searches (Multi-Path)
        phone = payload.get('phone_number')
        if phone:
            exact_hits, _ = self.qdrant.client.scroll(
                collection_name=self.qdrant.collection_name,
                scroll_filter=Filter(must=[FieldCondition(key="phone_number", match=MatchValue(value=phone))]),
                limit=5
            )
            for hit in exact_hits:
                if str(hit.id) not in candidates:
                    candidates[str(hit.id)] = {"payload": hit.payload, "vector_score": 0.5}
                    
        email = payload.get('email')
        if email:
            exact_hits, _ = self.qdrant.client.scroll(
                collection_name=self.qdrant.collection_name,
                scroll_filter=Filter(must=[FieldCondition(key="email", match=MatchValue(value=email))]),
                limit=5
            )
            for hit in exact_hits:
                if str(hit.id) not in candidates:
                    candidates[str(hit.id)] = {"payload": hit.payload, "vector_score": 0.5}

        # 3. Score Candidates
        results = []
        for cand_id, cand_data in candidates.items():
            score = EnsembleScorer.calculate_score(payload, cand_data['payload'], cand_data['vector_score'])
            logger.info(f"Candidate {cand_id}: vector={cand_data['vector_score']:.3f}, final_score={score:.3f}")
            if score >= settings.match_threshold:
                results.append({
                    "record_id": cand_id,
                    "score": score,
                    "profile": cand_data['payload']
                })
                
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
