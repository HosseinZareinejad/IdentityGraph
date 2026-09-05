from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Qdrant Settings
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "profiles"
    
    # Model Settings
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Ensemble Weights
    weight_vector: float = 0.4
    weight_name: float = 0.4
    weight_username: float = 0.2
    
    # Resolution Config
    # Legacy single-platform path (src/scorer.py + src/retriever.py). This was
    # 0.0, which made /resolve return every retrieved candidate as a "match";
    # 0.75 is the threshold resolution.py already used for the same scoring.
    match_threshold: float = 0.75

    # Phase 3 cross-platform fusion path. This one is a CALIBRATED PROBABILITY,
    # not an ad-hoc similarity, so it is directly interpretable: 0.70 was the
    # best-F1 operating point on the held-out test split (precision 0.82,
    # recall 0.95). Raise it when false positives are costlier than misses -
    # for a system that attributes accounts to real people, they usually are.
    fusion_match_threshold: float = 0.70
    fusion_model_path: str = "data/fusion_model.json"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Create a global settings instance
settings = Settings()
