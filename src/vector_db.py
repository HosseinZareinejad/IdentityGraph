from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from src.config import settings

class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(settings.qdrant_url)
        self.collection_name = settings.qdrant_collection
        # Initialize model lazy to avoid slow startup if not needed
        self.model = None

    def get_model(self):
        if self.model is None:
            self.model = SentenceTransformer(settings.embedding_model)
        return self.model

    def embed_text(self, text: str) -> list[float]:
        model = self.get_model()
        return model.encode(text).tolist()

    def insert_profile(self, record_id: str, payload: dict, name_text: str, city_text: str, bio_text: str):
        name_vec = self.embed_text(name_text)
        city_vec = self.embed_text(city_text)
        bio_vec = self.embed_text(bio_text)
        point = PointStruct(
            id=record_id,
            vector={
                "name": name_vec,
                "city": city_vec,
                "bio": bio_vec
            },
            payload=payload
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        return True
