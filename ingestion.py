import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
from hazm import Normalizer
import uuid
import sys
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

print("Loading dataset...")
df = pd.read_parquet('data/identity_dataset.parquet')
# Ensure no NaNs in string columns to avoid errors
df = df.fillna({
    'full_name': '',
    'bio': '',
    'city': '',
    'phone_number': '',
    'email': '',
    'username': ''
})

print("Initializing Hazm Normalizer...")
normalizer = Normalizer()

print("Loading Multilingual Sentence Transformer Model...")
# Use a multilingual model for Persian text
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Connecting to Qdrant...")
try:
    client = QdrantClient("http://127.0.0.1:6333", timeout=60.0)
    # Test connection
    client.get_collections()
except Exception as e:
    print(f"Failed to connect to Qdrant. Is the Docker container running? Error: {e}")
    sys.exit(1)

collection_name = "profiles"

# Create collection if it doesn't exist
try:
    client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists. Recreating...")
    client.delete_collection(collection_name)
except Exception:
    pass

# paraphrase-multilingual-MiniLM-L12-v2 has a vector size of 384
print(f"Creating Qdrant collection: {collection_name}")
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "name": VectorParams(size=384, distance=Distance.COSINE),
        "city": VectorParams(size=384, distance=Distance.COSINE),
        "bio": VectorParams(size=384, distance=Distance.COSINE)
    },
)

print("Creating Payload Indexes...")
client.create_payload_index(collection_name, field_name="city", field_schema=PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name, field_name="birth_year", field_schema=PayloadSchemaType.INTEGER)
client.create_payload_index(collection_name, field_name="phone_number", field_schema=PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name, field_name="email", field_schema=PayloadSchemaType.KEYWORD)

# Prepare data for insertion
print("Preparing data and generating embeddings in batches...")
batch_size = 128
points = []

# Qdrant requires IDs to be integers or UUID strings. 
# Our record_ids are UUID strings, which is perfect.

total_records = len(df)
for start_idx in range(0, total_records, batch_size):
    end_idx = min(start_idx + batch_size, total_records)
    batch_df = df.iloc[start_idx:end_idx]
    
    names = []
    cities = []
    bios = []
    
    for _, row in batch_df.iterrows():
        name = normalizer.normalize(row['full_name']) if row['full_name'] else "نامشخص"
        city = normalizer.normalize(row['city']) if row['city'] else "نامشخص"
        bio = normalizer.normalize(row['bio']) if row['bio'] else "بدون بیوگرافی"
        
        names.append(name)
        cities.append(city)
        bios.append(bio)
    
    # Generate embeddings for the batch
    name_embeddings = model.encode(names, show_progress_bar=False)
    city_embeddings = model.encode(cities, show_progress_bar=False)
    bio_embeddings = model.encode(bios, show_progress_bar=False)
    
    # Create PointStructs
    batch_points = []
    for i, (_, row) in enumerate(batch_df.iterrows()):
        payload = {
            "entity_id": str(row['entity_id']),
            "full_name": names[i],
            "username": row['username'],
            "email": row['email'],
            "bio": bios[i],
            "city": cities[i],
            "birth_year": int(row['birth_year']) if not pd.isna(row['birth_year']) else None,
            "phone_number": row['phone_number'],
            "is_noisy": bool(row['is_noisy'])
        }
        
        point = PointStruct(
            id=str(row['record_id']),
            vector={
                "name": name_embeddings[i].tolist(),
                "city": city_embeddings[i].tolist(),
                "bio": bio_embeddings[i].tolist()
            },
            payload=payload
        )
        batch_points.append(point)
        
    # Upsert batch to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=batch_points
    )
    print(f"Inserted {end_idx}/{total_records} records...")

print("Ingestion complete! Data successfully loaded into Qdrant.")
