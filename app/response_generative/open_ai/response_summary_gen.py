from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Optional

# Define prompt template once (reuse for all calls)
prompt_template = ChatPromptTemplate.from_template(
    """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    Context:
    {result_from_db}

    Question:
    {question}

    Answer:"""
)

def chat_summary(question: str, result_from_db: str) -> Optional[str]:
    """
    Generates a concise answer to a question using provided database results as context.

    Args:
        question (str): The user question to be answered.
        result_from_db (str): Textual content retrieved from the database, used as context.

    Returns:
        str | None: A summarized answer from the model or a fallback message if an error occurs.
    """
    try:
        # Load LLM once at module level (reuse in all requests)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
        # Format the prompt with actual inputs
        formatted_prompt = prompt_template.format_messages(
            question=question,
            result_from_db=result_from_db
        )

        # Get the model response
        response = llm.invoke(formatted_prompt)

        return response.content
    except Exception as e:
        print(f"Error in chat_summary: {e}")
        return "Something went wrong, please try again."
