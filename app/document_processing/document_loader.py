from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
from langchain.schema import Document
import os

def load_pdf(file_path: str) -> List[Document]:
    loader = PyMuPDFLoader(file_path)
    try:
        return loader.load()
    except Exception as e:
        raise Exception(f"Failed to load PDF {file_path}: {str(e)}") 
