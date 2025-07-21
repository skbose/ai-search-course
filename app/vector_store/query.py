# vector_store/query.py
from .qdrant_client import client
from qdrant_client.models import Filter, SearchParams
import os

def search_query(embedding, top_k=5):
    results = client.search(
        collection_name=os.getenv("QDRANT_COLLECTION"),
        query_vector=embedding,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128),
        with_payload=True
    )
    
    return results
