from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from src.config import settings

def prepare_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    return chunks

def build_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
    # Using Chroma in-memory for student project simplicity
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

def build_bm25(chunks):
    # Tokenize the page content for BM25
    corpus = [d.page_content.split() for d in chunks]
    bm25 = BM25Okapi(corpus)
    return bm25

def hybrid_search(query, vectorstore, bm25, chunks, k=settings.TOP_K):
    # 1. Semantic Search (Vector)
    semantic_docs = vectorstore.similarity_search(query, k=k)
    
    # 2. Keyword Search (BM25)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Pair chunks with their BM25 scores
    # We rely on 'chunks' list order matching the 'corpus' order used in build_bm25
    keyword_hits = sorted(
        zip(chunks, bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:k]
    
    # 3. Combine and Deduplicate (preserves order of semantic, appends keywords)
    # Using a dict to deduplicate by page_content
    combined = {}
    
    for doc in semantic_docs:
        combined[doc.page_content] = doc
        
    for doc, score in keyword_hits:
        if doc.page_content not in combined:
            combined[doc.page_content] = doc
            
    return list(combined.values())[:k]