from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Initialize once (reuse this instance)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

def generate_embeddings(chunks: List[Document]) -> List[List[float]]:
    """
    Generates embeddings from a list of LangChain Document objects.
    
    Args:
        chunks (List[Document]): List of text chunks.
    
    Returns:
        List[List[float]]: List of embedding vectors.
    """
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_model.embed_documents(texts)
    return embeddings

def generate_query_embedding(query: str) -> List[float]:
    """
    Generates an embedding vector for a search query.
    
    Args:
        query (str): The user's search query.
    
    Returns:
        List[float]: Embedding vector for the query.
    """
    return embedding_model.embed_query(query)
