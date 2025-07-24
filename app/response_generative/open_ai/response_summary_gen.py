from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def chat_summary(question, result_from_db):
    try:
        # Define prompt template
         prompt_template = PromptTemplate(
            input_variables=["question", "result_from_db"],
            template="""Use the following pieces of context to answer the question at the end.
                        If you don't know the answer, just say that you don't know, don't try to make up an answer.
                        Use three sentences maximum and keep the answer as concise as possible.
                        Always say "thanks for asking!" at the end of the answer.

                        Context:
                        {result_from_db}

                        Question:
                        {question}

                        Answer:"""
                        )
        
                  # Fill the prompt with actual values
         final_prompt = prompt_template.format(
            question=question,
            result_from_db=result_from_db)

          # Initialize LLM and call
         llm = ChatOpenAI(model_name="gpt-3.5 turbo", temperature=0.7)
         summary = llm.invoke(final_prompt)   
         return summary.content 
    except Exception as e:
      print("error in summaryPage:::",e)
    return "Something went wrong, please try again."