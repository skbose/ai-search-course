from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Initialize once (reuse this instance)
try:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
except Exception as e:
    print(f"❌ Failed to initialize OpenAI embeddings: {e}")
    print("💡 Make sure OPENAI_API_KEY is set in your environment")
    raise

def generate_embeddings(chunks: List[Document]) -> List[List[float]]:
    """
    Generate embedding vectors for a list of LangChain Document objects.
    
    Parameters:
        chunks (List[Document]): Documents whose text content will be embedded.
    
    Returns:
        List[List[float]]: Embedding vectors for each document's text. Returns an empty list if input is empty or if embedding generation fails.
    
    Raises:
        ValueError: If all documents contain empty or whitespace-only text.
    """
    if not chunks:
        return []
    
    texts = [doc.page_content for doc in chunks]
    
    if not any(text.strip() for text in texts):
        raise ValueError("All documents are empty. Cannot generate embeddings.")
    
    try: 
        embeddings = embedding_model.embed_documents(texts)
        return embeddings
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return []

def generate_query_embedding(query: str) -> List[float]:
    """
    Generate an embedding vector for the given search query string.
    
    Raises:
        ValueError: If the query is empty or contains only whitespace.
    
    Returns:
        A list of floats representing the embedding vector for the query.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    try:
         return embedding_model.embed_query(query)
    except Exception as e:
        print(f"❌ Failed to generate query embedding: {e}")
        raise
