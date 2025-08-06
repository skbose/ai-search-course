from app.response_generative.open_ai.response_summary_gen import chat_summary,extract_candidate_texts
from app.vector_store.query import search_query
import gradio as gr
from app.embeddings.generator import generate_query_embedding
from sentence_transformers import CrossEncoder

#load the cross-encoder model for ranking
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6-v2')
cross_encoder = CrossEncoder('cross-encoder/qnli-distilroberta-base')
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-electra-base')
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-roberta-base')
# cross_encoder = CrossEncoder('cross-encoder/nli-roberta-base')
# cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')
# cross_encoder = CrossEncoder('cross-encoder/contriever-msmarco')




# Chatbot logic
def chat_function(message, history):
    """
    Handles the main logic for responding to user queries in a conversational interface.
    
    Steps:
    1. Validates user input.
    2. Generates an embedding for the user's query using OpenAI's embedding model.
    3. Searches the vector database for the most similar context based on the embedding.
    4. Extracts relevant text from the vector DB result.
    5. Passes the context and user message to a summary generation function to create a response.
    
    Args:
        message (str): The latest message entered by the user.
        history (list): The previous chat history (not used in current logic, reserved for future use).
    
    Returns:
        list[dict]: A list containing a single response dictionary with role and content.
    """
    try:
        # ✅ Input validation
        if not message or len(message.strip()) == 0:
            return [{"role": "assistant", "content": "⚠️ Please enter something to continue."}]
        
        # 🔹 Step 1: Generate embedding
        embeddings = generate_jina_query_embedding(message)
        if not embeddings or not isinstance(embeddings, list):
            return [{"role": "assistant", "content": "⚠️ Failed to generate query embedding."}]

        
        # 🔹 Step 2: Search similar result with embedding
        result = search_query(embeddings,top_k=10)

        print(f"🔍 Search result: {result}")

        if not result or not isinstance(result, list) or not hasattr(result[0], 'payload'):
            return [{"role": "assistant", "content": "⚠️ No results found in vector store."}]
        
        # ✅ Extract candidate texts
        candidates = extract_candidate_texts(result)

        if not candidates:
            return [{"role": "assistant", "content": "⚠️ Retrieved content is empty or invalid."}]

        # 🔹 Step 3: Re-rank using cross-encoder
        top_k_contexts = rerank_with_cross_encoder(message, candidates, top_k=5)

        if not top_k_contexts:
            return [{"role": "assistant", "content": "⚠️ Failed to rerank retrieved texts."}]
    
        # ✅ Join top-k contexts into a single string
        combined_context = "\n\n".join(top_k_contexts)

        # ✅ Extract text from vector search payload
        # texts = result[0].payload.get('text')
        # if not texts or not isinstance(texts, str):
        #     return [{"role": "assistant", "content": "⚠️ Invalid content retrieved from vector DB."}]
     
        # 🔹 Step 4: Generate summary from context + user question
        summary = chat_summary(message, combined_context)
        if not summary or not isinstance(summary, str):
            return [{"role": "assistant", "content": "⚠️ Failed to generate response summary."}]

        return [{"role": "assistant", "content": summary}]
        
    except Exception as e:
        print(f"Error in chat_function: {e}")
        return [{"role": "assistant", "content": "⚠️ Something went wrong please try again."}]

# Gradio UI
def chat_with_model_ui():
    return gr.ChatInterface(
        fn=chat_function,
        title="💬 AI Chat — Text Chunker & Embedder",
        theme=gr.themes.Soft(primary_hue="indigo"),
        type="messages"
    )

# Reranking function
# This function is used to rerank the candidate texts based on their relevance to the query.
def rerank_with_cross_encoder(query, candidates, top_k=1):
    """
    Reranks candidate texts using a cross-encoder by relevance to the query.

    Args:
        query (str): The user input query.
        candidates (list[str]): List of context texts to rerank.
        top_k (int): Number of top results to return.

    Returns:
        list[str]: Top-k reranked texts sorted by relevance.
    """
    if not candidates or not isinstance(candidates, list):
        return []

    pairs = [(query, text) for text in candidates]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    
    # Extract top-k texts only
    return [text for _, text in ranked[:top_k]]
