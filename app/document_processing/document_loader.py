from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
from langchain.schema import Document

def load_pdf(file_path: str) -> Document:
    """
    Load a PDF file and return all its contents as a single Document object.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        Document: A single Document containing all the PDF text.

    Raises:
        Exception: If loading the PDF fails.
    """
    loader = PyMuPDFLoader(file_path)
    try:
        pages: List[Document] = loader.load()
        
        if not pages:
            raise Exception(f"No pages found in PDF: {file_path}")

        # Combine all page contents
        combined_text = "\n".join(page.page_content for page in pages)

        # Optionally combine metadata (you can customize this)
        combined_metadata = {"source": file_path, "total_pages": len(pages)}
        print(combined_metadata)
        return [Document(page_content=combined_text, metadata=combined_metadata)]
    
    except Exception as e:
        raise Exception(f"Failed to load PDF {file_path}: {str(e)}") from e
