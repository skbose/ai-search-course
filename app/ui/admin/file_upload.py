import gradio as gr
import os

# ✅ Import your pipeline functions here
from app.document_processing.document_loader import load_pdf
from app.document_processing.text_chunker import chunk_documents
from app.embeddings.generator import generate_embeddings
from app.vector_store.indexer import index_embeddings

def upload_file(file):
    status = []  # Collect status messages
    try:
        if not file or not file.name:
            return "❌ No file uploaded. Please upload a valid PDF file."
        
        file_path = file.name
        # validate file file extension
        if not file_path.lower().endswith('.pdf'):
            return "❌ Invalid file type. Please upload a PDF file."

        status.append(f"✅ File received: {os.path.basename(file_path)}")

        # ✅ Step 1: Load the PDF
        docs = load_pdf(file_path)
        status.append(f"✅ Loaded {len(docs)} page(s) from PDF.")
        
        # ✅ Step 2: Chunk the documents
        chunks = chunk_documents(docs, 1000, 200)
        status.append(f"✅ Created {len(chunks)} chunks.")
        
        # ✅ Step 3: Preview all chunk contents
        status.append("📄 All chunks content:")
        for i, chunk in enumerate(chunks):
            # content_preview = chunk.page_content.strip()[:300]  # show only first 300 chars
            status.append(f"\n--- Chunk {i + 1} ---\n{chunk.page_content}")

        # ✅ Step 4: Generate embeddings
        embeddings = generate_embeddings(chunks)
        status.append(f"✅ Generated {len(embeddings)} embeddings.")
        status.append("🔢 Sample embedding (first 10 values): " + str(embeddings[0][:10]))

        # ✅ Step 5: Index into Qdrant
        texts = [chunk.page_content for chunk in chunks]
        index_embeddings(embeddings, texts)
        status.append("📥 Embeddings indexed into Qdrant successfully.")

        return "\n\n".join(status)

    except Exception as e:
        import traceback
        error_msg = f"❌ Failed to process: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        status.append(error_msg)
        return "\n\n".join(status)


def get_upload_ui():
    return gr.Interface(
        fn=upload_file,
        inputs=gr.File(file_types=[".pdf"]),
        outputs="text",
        title="Upload PDF and Index"
    )
