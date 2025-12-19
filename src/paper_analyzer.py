from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from src.config import settings

def clean_text(text: str) -> str:
    """
    Removes surrogate characters that crash the embedding model.
    This encodes to UTF-8 ignoring errors, then decodes back to string.
    """
    return text.encode('utf-8', 'ignore').decode('utf-8')

def load_and_extract(pdf_path: str) -> dict:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # --- FIX: Clean every page immediately ---
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    # -----------------------------------------

    full_text = " ".join([d.page_content for d in docs])
    
    return {
        "raw_documents": docs,
        "full_text": full_text
    }

def extract_structure(text: str) -> str:
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.MODEL_NAME,
        temperature=0
    )
    
    prompt = f"""
    Analyze the following research paper text and extract:
    - Title
    - Authors
    - Research area
    - Main contributions (bullet points)
    - Methodology summary
    - Results summary
    - Conclusion

    Text:
    {text[:10000]} 
    """
    
    return llm.invoke(prompt).content