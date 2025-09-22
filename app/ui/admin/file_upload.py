import gradio as gr
import os, time, json

# ✅ Import your pipeline functions here
from app.document_processing.document_loader import load_pdf
from app.document_processing.text_chunker import token_chunk_documents
from app.embeddings.generator import get_token_embeddings_local
from app.vector_store.indexer import index_embeddings
from app.mlflow_utils import start_experiment, log_params, log_metrics, log_artifact

def upload_file(file):
    """
    Processes an uploaded PDF file by validating, chunking, generating embeddings, and indexing them, returning a detailed status message.
    
    The function validates the uploaded file to ensure it is a PDF, loads its content, splits it into text chunks, generates embeddings for each chunk, and indexes the embeddings into a Qdrant vector store. It returns a concatenated status message summarizing each processing step, including previews of chunk content and sample embedding data. If an error occurs, a detailed error message with traceback is returned.
    """
    status = []  # Collect status messages
    try:
        if not file or not file.name:
            return "❌ No file uploaded. Please upload a valid PDF file."
        
        file_path = file.name
        # validate file file extension
        if not file_path.lower().endswith('.pdf'):
            return "❌ Invalid file type. Please upload a PDF file."
        
        # Start mlflow run
        with start_experiment(run_name=f"document_upload_{os.path.basename(file_path)}"):
            log_params({
                "file_name":os.path.basename(file_path),
                "embedding_model":"jina-embeddings-v2-base-en",
                "chunk_size":6000,
                "chunk_overlap":500,
                "vector_db":"qdrant"
            })

        status.append(f"✅ File received: {os.path.basename(file_path)}")

        # ✅ Step 1: Load the PDF
        t0 = time.time()
        docs = load_pdf(file_path)
        log_metrics({"load_time_sec":time.time()-t0,"num_pages":len(docs)})
        status.append(f"✅ Loaded {len(docs)} page(s) from PDF.")
        
        # ✅ Step 2: Chunk the documents 
        # chunks = chunk_documents(docs, 1000, 200)  # using character-based chunking
        chunks = token_chunk_documents(docs, 6000, 500) # using token-based chunking
        log_metrics({"num_chunks":len(chunks)})
        status.append(f"✅ Created {len(chunks)} chunks.")
        
        
        # ✅ Step 3: Preview all chunk contents
        status.append("📄 All chunks content:")
        for i, chunk in enumerate(chunks):
            # content_preview = chunk.page_content.strip()[:300]  # show only first 300 chars
            status.append(f"\n--- Chunk {i + 1} ---\n{chunk.page_content}")

        #save preview artifact
        preview_path = "data/chunks_preview.json"
        with open(preview_path,"w") as f:
            json.dump([c.page_content[:300] for c in chunks],f)
        log_artifact(preview_path)
        
        # ✅ Step 4: Generate embeddings
        embeddings, metadatas = get_token_embeddings_local(chunks, window_size=50)
        log_metrics({"num_embeddings":len(embeddings)})
        status.append(f"✅ Generated {len(embeddings)} embeddings.")
        status.append("🔢 Sample embedding (first 10 values): " + str(embeddings[0][:10]))
        
        status.append("🔍 metadata: " + str(metadatas))
        
        # ✅ Step 5: Index into Qdrant
        # texts = [chunk.page_content for chunk in chunks]
        index_embeddings(embeddings, metadatas)
        status.append("📥 Embeddings indexed into Qdrant successfully.")
        
        return "\n\n".join(status)

    except Exception as e:
        import traceback
        error_msg = f"❌ Failed to process: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        status.append(error_msg)
        return "\n\n".join(status)


def get_upload_ui():
    """
    Create and return a Gradio interface for uploading and indexing PDF files.
    
    The interface allows users to upload a PDF, processes the file through a document pipeline, and displays status messages for each processing step.
    """
    return gr.Interface(
        fn=upload_file,
        inputs=gr.File(file_types=[".pdf"]),
        outputs="text",
        title="Upload PDF and Index"
    )
