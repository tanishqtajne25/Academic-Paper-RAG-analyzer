import streamlit as st
import os
import tempfile
from src.paper_analyzer import load_and_extract, extract_structure
from src.rag_system import prepare_chunks, build_vectorstore, build_bm25, hybrid_search
from src.chains import qa_chain

st.set_page_config(layout="wide", page_title="Research RAG")

st.title("📚 Multi-Paper Research Assistant")
st.markdown("---")

# Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "paper_structures" not in st.session_state: # Changed to list
    st.session_state.paper_structures = []

# Sidebar: Upload
with st.sidebar:
    st.header("1. Knowledge Base")
    # CHANGED: Added accept_multiple_files=True
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Analyze Papers"):
        
        # Reset state for new analysis
        st.session_state.paper_structures = []
        st.session_state.chunks = []
        all_raw_docs = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing: {uploaded_file.name}...")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # 1. Extraction
            data = load_and_extract(tmp_path)
            all_raw_docs.extend(data["raw_documents"]) # Collect docs for RAG
            
            # 2. Structure Extraction (LLM) - Stored with filename
            struct = extract_structure(data["full_text"])
            st.session_state.paper_structures.append({
                "filename": uploaded_file.name,
                "analysis": struct
            })
            
            # Cleanup temp file
            os.remove(tmp_path)
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Building Search Index (Embeddings + BM25)...")
        
        # 3. RAG Setup (Done once for ALL docs combined)
        if all_raw_docs:
            chunks = prepare_chunks(all_raw_docs)
            st.session_state.chunks = chunks
            st.session_state.vectorstore = build_vectorstore(chunks)
            st.session_state.bm25 = build_bm25(chunks)
            
            st.success(f"Processed {len(uploaded_files)} papers successfully!")
            progress_bar.empty()
            status_text.empty()

# Main Area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("2. Paper Summaries")
    if st.session_state.paper_structures:
        for p in st.session_state.paper_structures:
            with st.expander(f"📄 Analysis: {p['filename']}"):
                st.markdown(p['analysis'])
    else:
        st.info("Upload papers to see their structured summary.")

with col2:
    st.header("3. Chat with Knowledge Base")
    query = st.text_input("Ask a question across all papers:")
    
    if query and st.session_state.vectorstore:
        with st.spinner("Searching across all papers..."):
            # Hybrid Search
            retrieved_docs = hybrid_search(
                query, 
                st.session_state.vectorstore, 
                st.session_state.bm25, 
                st.session_state.chunks
            )
            
            # Answer Generation
            answer = qa_chain(query, retrieved_docs)
            
            st.markdown(f"**Answer:** {answer}")
            
            st.markdown("---")
            st.subheader("Sources")
            for i, doc in enumerate(retrieved_docs):
                # Show which paper the chunk came from
                source_src = doc.metadata.get('source', 'Unknown')
                # Clean up temp filename to show something readable if possible
                # (Note: PyPDFLoader puts the temp path in metadata, 
                # strictly speaking you'd need extra logic to map back to original filename, 
                # but for a student project, raw path is usually fine or we can just ignore it)
                
                with st.expander(f"Source {i+1} (from PDF)"):
                    st.caption(f"File path: {source_src}")
                    st.write(doc.page_content)