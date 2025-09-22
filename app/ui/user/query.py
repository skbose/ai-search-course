from app.response_generative.open_ai.response_summary_gen import chat_summary,extract_candidate_texts
from app.vector_store.query import search_query
import gradio as gr
from app.embeddings.generator import generate_jina_query_embedding
from sentence_transformers import CrossEncoder
from app.mlflow_utils import start_experiment, log_artifact, log_metrics, log_params
import time,json

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
        
        # Start mlflow run
        with start_experiment("ChatbotQueries", run_name="user_query"):
            log_params({
                "llm_model":"gpt-5-nano",
                "embedding_model": "jina-embeddings-v2-base-en",
                "cross_encoder": "qnli-distilroberta-base",
                "top_k_retrieval": 10,
                "top_k_rerank": 5
            })
        
        # 🔹 Step 1: Generate embedding
        t0 = time.time()
        embeddings = generate_jina_query_embedding(message)
        log_metrics({"embedding_time_sec":time.time()-t0})

        if not embeddings or not isinstance(embeddings, list):
            return [{"role": "assistant", "content": "⚠️ Failed to generate query embedding."}]

        
        # 🔹 Step 2: Search similar result with embedding
        t0 = time.time()
        result = search_query(embeddings,top_k=10)
        log_metrics({"search_time_sec": time.time() - t0, "search_results": len(result)})

        print(f"🔍 Search result: {result}")

        if not result or not isinstance(result, list) or not hasattr(result[0], 'payload'):
            return [{"role": "assistant", "content": "⚠️ No results found in vector store."}]
        
        # ✅ Extract candidate texts
        candidates = extract_candidate_texts(result)
        log_metrics({"num_candidates": len(candidates)})

        # Save candidates as artifact
        candidates_path = "data/candidates.json"
        with open(candidates_path, "w") as f:
            json.dump(candidates, f)
        log_artifact(candidates_path)

        if not candidates:
            return [{"role": "assistant", "content": "⚠️ Retrieved content is empty or invalid."}]

        # 🔹 Step 3: Re-rank using cross-encoder
        t0 = time.time()
        top_k_contexts = rerank_with_cross_encoder(message, candidates, top_k=5)
        log_metrics({"rerank_time_sec": time.time() - t0, "rerank_results": len(top_k_contexts)})


        if not top_k_contexts:
            return [{"role": "assistant", "content": "⚠️ Failed to rerank retrieved texts."}]
    
        # ✅ Join top-k contexts into a single string
        combined_context = "\n\n".join(top_k_contexts)


        # Save reranked contexts as artifact
        reranked_path = "data/reranked.json"
        with open(reranked_path, "w") as f:
            json.dump(top_k_contexts, f)
        log_artifact(reranked_path)
        
        print(f"🔍 Re-ordered context: {top_k_contexts}")

        # ✅ Extract text from vector search payload
        # texts = result[0].payload.get('text')
        # if not texts or not isinstance(texts, str):
        #     return [{"role": "assistant", "content": "⚠️ Invalid content retrieved from vector DB."}]
     
        # 🔹 Step 4: Generate summary from context + user question
        t0 = time.time()
        summary = chat_summary(message, combined_context)
        print(f"📝 Summary: {summary}")
        log_metrics({"response_time_sec": time.time() - t0})

        if not summary or not isinstance(summary, str):
            return [{"role": "assistant", "content": "⚠️ Failed to generate response summary."}]
        
        # Save final answer as artifact
        answer_path = "data/answer.json"
        with open(answer_path, "w") as f:
            json.dump({"query": message, "answer": summary}, f)
        log_artifact(answer_path)

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
