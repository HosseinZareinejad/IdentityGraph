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
    match_threshold: float = 0.0
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Create a global settings instance
settings = Settings()
