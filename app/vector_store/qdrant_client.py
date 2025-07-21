from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import os

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_docs")

client = QdrantClient(url=QDRANT_HOST)

def ensure_collection(vector_size: int):
    """Creates the collection if it doesn't exist."""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"📦 Collection already exists: {COLLECTION_NAME}")
