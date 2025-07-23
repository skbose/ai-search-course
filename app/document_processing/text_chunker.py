# text splitter / chunker for document processing

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
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
