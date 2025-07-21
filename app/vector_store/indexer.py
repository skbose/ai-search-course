from qdrant_client.models import Distance, VectorParams, PointStruct
from .qdrant_client import client, ensure_collection
import os
import uuid
from dotenv import load_dotenv
load_dotenv()

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "my_docs")



def index_embeddings(embeddings: list[list[float]], texts: list[str]):
    """
    Indexes the embedding vectors and their corresponding texts into Qdrant.
    """
    assert len(embeddings) == len(texts), "Mismatch between embeddings and texts"

    # Create points with metadata
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": text}
        )
        for embedding, text in zip(embeddings, texts)
    ]

    vector_size = len(embeddings[0])
    ensure_collection(vector_size)

    # Upload to Qdrant
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )

    print(f"✅ Indexed {len(points)} embeddings into Qdrant.")
