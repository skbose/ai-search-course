from app.response_generative.open_ai.response_summary_gen import chat_summary
from app.vector_store.query import search_query
import gradio as gr
from app.embeddings.generator import generate_jina_query_embedding



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
        if not result or not isinstance(result, list) or not hasattr(result[0], 'payload'):
            return [{"role": "assistant", "content": "⚠️ No results found in vector store."}]

        print(result)
        
        # ✅ Extract text from vector search payload
        texts = result[0].payload.get('text')
        if not texts or not isinstance(texts, str):
            return [{"role": "assistant", "content": "⚠️ Invalid content retrieved from vector DB."}]
     
        
        # 🔹 Step 3: Generate summary from context + user question
        summary = chat_summary(message, texts)
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
