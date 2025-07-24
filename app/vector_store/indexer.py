from qdrant_client.models import PointStruct
from .qdrant_client import client, ensure_collection, COLLECTION_NAME
import uuid


def index_embeddings(embeddings: list[list[float]], texts: list[str]):
    """
    Indexes a list of embedding vectors and their associated texts into a Qdrant vector database.
    
    Parameters:
        embeddings (list[list[float]]): Embedding vectors to be indexed.
        texts (list[str]): Texts corresponding to each embedding vector.
    
    Raises:
        ValueError: If either embeddings or texts is empty.
        AssertionError: If the number of embeddings does not match the number of texts.
    """
    
    if not embeddings or not texts:
        raise ValueError("Embeddings and texts cannot be empty")
    
    assert len(embeddings) == len(texts), "Mismatch between embeddings and texts"

    # Create points with metadata
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": text}
        )
        for embedding, text in zip(embeddings, texts, strict=True)
    ]

    vector_size = len(embeddings[0])
    ensure_collection(vector_size)

    # Upload to Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"✅ Indexed {len(points)} embeddings into Qdrant.")
