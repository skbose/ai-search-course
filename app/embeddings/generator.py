from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings  # For newer versions of LangChain
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
import torch
from langchain.schema import Document



# Initialize once (reuse this instance)
try:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    jina_embedding_model = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")  # or "jina-embeddings-v2-small-en"
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



# Load Jina model locally
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
model.eval()  # Disable dropout etc.

@torch.no_grad()
def get_token_embeddings_local(
    chunks: List[Document], window_size: int = 200
) -> Tuple[List[List[float]], List[dict]]:
    """
    Generate mean pooled token embeddings for text chunks using a local transformer model.

    This function processes each chunk of a document using a tokenizer and transformer model
    to obtain token-level embeddings. It then applies late chunking by sliding a window over
    the token embeddings and computes the mean embedding for each window. The corresponding
    decoded text of each window is also collected as metadata.

    Parameters:
        chunks (List[Document]): A list of LangChain Document objects containing the text to process.
        window_size (int, optional): The number of tokens per window for late chunking. 
                                     Defaults to 200.

    Returns:
        Tuple[List[List[float]], List[str]]:
            - A list of mean pooled embeddings (each is a list of floats) for each token window.
            - A list of corresponding decoded text strings for each token window (used as metadata).
    """
    chunk_vectors = []
    metadata_list = []

    for i, doc in enumerate(chunks):
        inputs = tokenizer(doc.page_content, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].squeeze(0)  # [seq_len]
        outputs = model(**inputs)
        
        # Get token embeddings [1, seq_len, hidden_dim]
        token_embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, 768]

        for i in range(0, token_embeddings.shape[0], window_size):
            window = token_embeddings[i:i+window_size]
            token_ids_window = input_ids[i:i + window_size]
            if window.shape[0] == 0:
                continue
            
            # ✅ Decode the token IDs to get the actual text for this window
            window_text = tokenizer.decode(token_ids_window, skip_special_tokens=True)
            
            # Add a conditional check to skip empty chunks
            if window_text:
                pooled_vector = window.mean(dim=0).tolist()
                chunk_vectors.append(pooled_vector)
                metadata_list.append(window_text)

    return chunk_vectors, metadata_list


def generate_jina_query_embedding(query: str) -> List[float]:
    """
    Generate an embedding vector for the given search query using the Jina model.

    Parameters:
        query (str): The search query string.

    Returns:
        List[float]: A list of floats representing the embedding vector for the query.

    Raises:
        ValueError: If the query is empty or contains only whitespace.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    try:
        # The JinaEmbeddings class from LangChain has a method called 'embed_query'
        # which is designed for this purpose.
        return jina_embedding_model.embed_query(query)
    except Exception as e:
        print(f"❌ Failed to generate Jina query embedding: {e}")
        raise