# text splitter / chunker for document processing

from langchain.text_splitter import RecursiveCharacterTextSplitter
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
