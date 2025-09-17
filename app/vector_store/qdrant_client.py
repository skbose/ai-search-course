from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import os

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_docs")

try:
    client = QdrantClient(url=QDRANT_HOST)
    # test connection
    client.get_collections()
    print(f"✅ Connected to Qdrant at {QDRANT_HOST}")
except Exception as e:
    print(f"❌ Failed to connect to Qdrant at {QDRANT_HOST}: {e}")
    raise

def ensure_collection(vector_size: int):
    """
    Ensure that a Qdrant collection with the specified vector size exists, creating it if necessary.
    
    Parameters:
        vector_size (int): The dimensionality of the vectors to be stored in the collection. Must be a positive integer.
    
    Raises:
        ValueError: If `vector_size` is not positive.
        Exception: If an error occurs while checking or creating the collection.
    """
    if vector_size <= 0:
        raise ValueError(f"Vector size must be positive, got {vector_size}")
    
    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"✅ Created collection: {COLLECTION_NAME}")
        else:
            print(f"📦 Collection already exists: {COLLECTION_NAME}")
    except Exception as e:
        print(f"❌ Failed to ensure collection {COLLECTION_NAME}: {e}")
        raise