from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Optional
from guardrails import Guard, OnFailAction
from app.guardrails.validators import NoOffensiveLanguage,NoFalseClaims,NoPersonalInfo,NoVerbositySpam,EndsWithThanks,NoRepetition
from guardrails.hub import RegexMatch, ProfanityFree, DetectPII 

guard = (
    Guard()
    # Hub validators
    # .use(ProfanityFree(on_fail=OnFailAction.FIX))
    # .use(DetectPII(on_fail=OnFailAction.FIX))

    # Custom validators
    .use(NoFalseClaims(on_fail=OnFailAction.EXCEPTION))
    .use(NoOffensiveLanguage(on_fail=OnFailAction.EXCEPTION))
    .use(NoPersonalInfo(on_fail=OnFailAction.EXCEPTION))
    .use(NoVerbositySpam(on_fail=OnFailAction.EXCEPTION))
    .use(NoRepetition(on_fail=OnFailAction.EXCEPTION))
)
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
        # Load LLM once at module level (reuse in all requests)gpt-5-nano
        llm = ChatOpenAI(model_name="gpt-5-nano")
        
        # Format the prompt with actual inputs
        formatted_prompt = prompt_template.format_messages(
            question=question,
            result_from_db=result_from_db
        )

        # Get the model response
        response = llm.invoke(formatted_prompt)

        if not response or not getattr(response, "content", None):
            print("⚠️ LLM returned empty response.")
            return None
        
        try:
            # Guard validation
            validated_output = guard.parse(response.content, llm=llm)

            print("🔹 Raw LLM response:", response.content)
            print("validated_output:", validated_output)

            cleaned = getattr(validated_output, "validated_output", None)

            # Special-case PII override
            if validated_output and getattr(validated_output, "validation_summaries", None):
                for summary in validated_output.validation_summaries:
                    if summary.validator_name == "DetectPII" and summary.validator_status == "fail":
                        cleaned = "Response contains personal information. So cannot be displayed. thanks for asking!"

            if not cleaned:
                print("⚠️ No validated_output, falling back to raw LLM output")
                cleaned = getattr(validated_output, "raw_llm_output", None)

            return cleaned.strip() if cleaned else None

        except Exception as ve:
            # This block runs if Guard validator raises (OnFailAction.EXCEPTION)
            print(f"❌ Validation failed: {ve}")
            return str(ve)  # return the validator's error message
    
    except Exception as e:
        print(f"Error in chat_summary: {e}")
        return "Something went wrong, please try again."
    
def extract_candidate_texts(results):
    """
    Extracts valid non-empty text entries from the payloads of vector DB results.

    Args:
        results (list): List of vector DB search results.

    Returns:
        list[str]: A list of valid text strings extracted from payloads.
    """
    candidates = []
    for item in results:
        if hasattr(item, 'payload'):
            text = item.payload.get('text')
            if isinstance(text, str) and len(text.strip()) > 0:
                candidates.append(text)
    return candidates
