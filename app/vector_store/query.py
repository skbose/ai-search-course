# vector_store/query.py
from .qdrant_client import client
from typing import List, Any
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import SearchParams
import os

def search_query(embedding: List[float], top_k: int = 5) -> List[ScoredPoint]:
    if not embedding or not isinstance(embedding, list):
        raise ValueError("Embedding must be a non-empty list of floats")
    
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    collection_name = os.getenv("QDRANT_COLLECTION")
    if not collection_name:
        raise ValueError("QDRANT_COLLECTION environment variable not set")
    
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=128),
            with_payload=True
        )
        return results
    except Exception as e:
        raise Exception(f"Vector search failed: {str(e)}")