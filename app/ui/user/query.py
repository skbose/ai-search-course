from app.response_generative.open_ai.response_summary_gen import chat_summary
from app.vector_store.query import search_query
import gradio as gr
from app.embeddings.generator import generate_query_embedding



# Chatbot logic
def chat_function(message, history):
    try:
        if not message or len(message.strip()) == 0:
            return [{"role": "assistant", "content": "⚠️ Please enter something to continue."}]
        
         # 🔹 Step 1: Generate embeddings for the chunks
        embeddings = generate_query_embedding(message)
        result = search_query(embeddings,top_k=1)
        texts = result[0].payload['text']      
        return [{"role": "assistant", "content": texts}]
      # summary = chat_summary(message, texts)
        # return [{"role": "assistant", "content": summary}]
        
    except Exception as e:
        # print('error::::>>>>',e)
        return [{"role": "assistant", "content": "⚠️ Something went wrong please try again."}]

# Gradio UI
def chat_with_model_ui():
    return gr.ChatInterface(
        fn=chat_function,
        title="💬 AI Chat — Text Chunker & Embedder",
        theme=gr.themes.Soft(primary_hue="indigo"),
        type="messages"
    )
