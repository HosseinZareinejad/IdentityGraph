from rapidfuzz.distance import JaroWinkler, Levenshtein
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class EnsembleScorer:
    @staticmethod
    def calculate_score(source: dict, cand: dict, vector_score: float) -> float:
        # 1. Name Similarity
        s_name = source.get('full_name', '')
        c_name = cand.get('full_name', '')
        name_score = JaroWinkler.normalized_similarity(s_name, c_name) if s_name and c_name else 0.0
        
        # 2. Username Similarity
        s_user = source.get('username', '')
        c_user = cand.get('username', '')
        user_score = Levenshtein.normalized_similarity(s_user, c_user) if s_user and c_user else 0.0
        
        # 3. Exact Match Boosts
        s_phone = source.get('phone_number', '')
        c_phone = cand.get('phone_number', '')
        if s_phone and c_phone and s_phone == c_phone:
            return 1.0
            
        s_email = source.get('email', '')
        c_email = cand.get('email', '')
        if s_email and c_email and s_email == c_email:
            return 1.0
            
        # 4. Age Penalty
        penalty = 0.0
        s_year = source.get('birth_year')
        c_year = cand.get('birth_year')
        if s_year and c_year:
            try:
                diff = abs(int(s_year) - int(c_year))
                if diff > 5:
                    penalty -= 0.3
            except ValueError:
                pass
                
        # Weighted Ensemble
        final_score = (
            (vector_score * settings.weight_vector) + 
            (name_score * settings.weight_name) + 
            (user_score * settings.weight_username) + 
            penalty
        )
        
        return max(0.0, min(1.0, final_score))
