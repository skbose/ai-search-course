# vector_store/query.py
from .qdrant_client import client
from typing import List
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import SearchParams
import os

def search_query(embedding: List[float], top_k: int = 5) -> List[ScoredPoint]:
    """
    Perform a vector similarity search using the provided embedding and return the top matching results.
    
    Parameters:
        embedding (List[float]): The embedding vector to search with.
        top_k (int, optional): The number of top results to return. Must be positive. Defaults to 5.
    
    Returns:
        List[ScoredPoint]: A list of scored points representing the most similar vectors found.
    
    Raises:
        ValueError: If the embedding is not a non-empty list, if top_k is not positive, or if the QDRANT_COLLECTION environment variable is not set.
        Exception: If the vector search fails for any other reason.
    """
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
        raise Exception(f"Vector search failed: {str(e)}") from e