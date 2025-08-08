# text splitter / chunker for document processing

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from typing import List
from langchain.schema import Document

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split a list of Document objects into smaller overlapping text chunks.
    
    Parameters:
        documents (List[Document]): The list of Document objects to be chunked.
        chunk_size (int, optional): Maximum number of characters in each chunk. Defaults to 1000.
        chunk_overlap (int, optional): Number of overlapping characters between consecutive chunks. Defaults to 200.
    
    Returns:
        List[Document]: A list of Document objects representing the resulting text chunks.
    
    Raises:
        ValueError: If the documents list is empty, chunk_size is not positive, chunk_overlap is negative, or chunk_overlap is greater than or equal to chunk_size.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")
    
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative")
    
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "."]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def token_chunk_documents(documents: List[Document], chunk_size: int = 6000, chunk_overlap: int = 500) -> List[Document]:
    """
    Splits documents into chunks based on token count using a token-based splitter.

    Args:
        documents (List[Document]): The list of LangChain Document objects to split.
        chunk_size (int): Approximate maximum tokens per chunk (default is 6000).
        chunk_overlap (int): Overlapping tokens between chunks (default is 500).

    Returns:
        List[Document]: A list of token-based chunks.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size")

    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # encoding_name="cl100k_base"  # same encoding as OpenAI / Jina
    )

    return text_splitter.split_documents(documents)
