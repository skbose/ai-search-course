import gradio as gr
import shutil
from pathlib import Path

# ✅ Set dynamic upload directory relative to current script
BASE_DIR = Path(__file__).resolve().parent  # e.g., app/ui/admin
UPLOAD_DIR = BASE_DIR.parents[1] / "uploaded_documents"  # e.g., app/uploaded_documents
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Import your pipeline functions
from app.document_processing.document_loader import load_pdf
from app.document_processing.text_chunker import chunk_documents
# from app.embeddings.generator import generate_embeddings
# from app.vector_store.indexer import index_embeddings

def upload_file(file):
    status = []  # Collect status messages

    try:
        if not file or not file.name:
            return "❌ No file uploaded. Please upload a valid PDF file."

        file_path = file.name
        file_name = Path(file_path).name

        # ✅ Check if file already exists
        save_path = UPLOAD_DIR / file_name
        if save_path.exists():
            return f"⚠️ The file '{file_name}' has already been uploaded."

        # ✅ Check file extension
        if not file_name.lower().endswith('.pdf'):
            return "❌ Invalid file type. Please upload a PDF file."

        # ✅ Save the file
        shutil.copy(file_path, save_path)

        status.append(f"✅ File received: {file_name}")

        # ✅ Step 1: Load the PDF
        docs = load_pdf(save_path)
        status.append(f"✅ Loaded {len(docs)} page(s) from PDF.")

        # ✅ Step 2: Chunk the documents
        chunks = chunk_documents(docs, 1000, 200)
        status.append(f"✅ Created {len(chunks)} chunks.")

        # ✅ Step 3: Preview chunk contents
        status.append("📄 All chunks content:")
        for i, chunk in enumerate(chunks):
            status.append(f"\n--- Chunk {i + 1} ---\n{chunk.page_content}")

        # ✅ Step 4 and 5: Embedding (optional)
        # embeddings = generate_embeddings(chunks)
        # status.append(f"✅ Generated {len(embeddings)} embeddings.")
        # status.append("🔢 Sample embedding (first 10 values): " + str(embeddings[0][:10]))

        # texts = [chunk.page_content for chunk in chunks]
        # index_embeddings(embeddings, texts)
        # status.append("📥 Embeddings indexed into Qdrant successfully.")

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
