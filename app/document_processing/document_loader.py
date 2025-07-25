from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
from langchain.schema import Document

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file from the specified path and return its contents as a list of Document objects.
    
    Parameters:
        file_path (str): The path to the PDF file to be loaded.
    
    Returns:
        List[Document]: A list of Document objects representing the contents of the PDF.
    
    Raises:
        Exception: If the PDF cannot be loaded for any reason.
    """
    loader = PyMuPDFLoader(file_path)
    try:
        return loader.load()
    except Exception as e:
        raise Exception(f"Failed to load PDF {file_path}: {str(e)}") from e
