# main.py
from dotenv import load_dotenv
load_dotenv()


from pathlib import Path
from document_processing.document_loader import load_pdf
from document_processing.text_chunker import chunk_documents
from embeddings.generator import generate_embeddings, generate_query_embedding
from vector_store.indexer import index_embeddings
from vector_store.query import search_query


if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent.parent / "data" / "sample1.pdf"
    
    # Step 1: Load the PDF
    docs = load_pdf(file_path)
    print(f"✅ Loaded {len(docs)} page(s) from PDF.")
    
    # # Step 2: Chunk the documents
    chunks = chunk_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")

    # # Step 3: Preview a sample chunk
    print("📄 Sample chunk content:\n", chunks[0].page_content[:300])

    # # Step 4: Generate embeddings
    embeddings = generate_embeddings(chunks)
    print(f"✅ Generated {len(embeddings)} embeddings.")
    print("🔢 Sample embedding (first 10 values):", embeddings[0][:10])

    # # Step 5: Index into Qdrant
    index_embeddings(embeddings, chunks)
    print("📥 Embeddings indexed into Qdrant successfully.")
    

    while True:
        query = input("\n❓ Enter your question (or type 'exit' to quit): ").strip()
        
        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        embedding = generate_query_embedding(query)
        results = search_query(embedding, 2)

        if not results:
            print("⚠️ No results found.")
            continue
        
        print(f"🔍 Found {len(results)} results for your query:\n")
        if len(results) > 0:
            print("Full payload:", results[0].payload)

        for i, res in enumerate(results, 1):
            print(f"\n🔹 Result {i}")
            print("Score:", res.score)
            print("Text:", res.payload.get("page_content", "")[:300], "...")
