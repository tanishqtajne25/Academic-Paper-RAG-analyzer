from langchain_groq import ChatGroq # <-- CHANGED
from src.config import settings

def qa_chain(query, contexts):
    # Initialize Groq
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.MODEL_NAME,
        temperature=0.2
    )
    
    context_text = "\n\n---\n\n".join([c.page_content for c in contexts])
    
    prompt = f"""
    You are a Research Assistant. Answer the question ONLY using the provided Context.
    If the answer is not in the context, say "Not found in the paper context."

    Context:
    {context_text}

    Question: {query}
    
    Answer:
    """
    
    response = llm.invoke(prompt)
    return response.content