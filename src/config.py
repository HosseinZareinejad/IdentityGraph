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

    # Cross-platform fusion path. This one is a CALIBRATED PROBABILITY, not an
    # ad-hoc similarity, so it is directly interpretable.
    #
    # Chosen by Phase 6's protocol, not by a sweep over the reported data.
    # On held-out validation entities the end-to-end clustering objective is
    # FLAT from 0.30 to 0.75 (B-cubed F1 0.8883 down to 0.8844, against a
    # standard error of 0.0045), so the argmax inside it is noise. The tie is
    # broken one stage further down, on the objective the system actually
    # exists to serve - mapping a cluster to the right real person - where
    # 0.30 wins by a margin that is consistent rather than large: top-1 82.8%
    # vs 82.0%, and 72.4% vs 70.7% on the twin cases.
    #
    # The direction makes sense: a stricter threshold buys cluster purity by
    # splitting clusters, and a split cluster loses exactly the pooled
    # evidence - the LinkedIn job, the Telegram phone - that the mapper needs.
    fusion_match_threshold: float = 0.30
    fusion_model_path: str = "data/fusion_model.json"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Create a global settings instance
settings = Settings()
