# academic-rag-analyzer

A comprehensive research assistant tool designed to streamline the analysis of academic literature. This application leverages a Hybrid RAG (Retrieval-Augmented Generation) architecture to provide accurate, context-aware answers across multiple PDF documents simultaneously.

## Features

### 1. Hybrid Search Architecture
Most RAG systems rely solely on semantic vector search, which can miss specific acronyms or exact phrasing common in research papers. This project implements a **Hybrid Search** strategy:
* **Vector Search (Dense):** Uses local `llama3.1` embeddings via Ollama stored in ChromaDB to capture conceptual meaning.
* **BM25 (Sparse):** Parallel keyword indexing using `rank_bm25` to capture exact matches and technical terminology.
* **Deduplication:** A custom algorithm merges results from both indexes, prioritizing semantic matches while ensuring unique context for the LLM.

### 2. Automated Structured Analysis
Upon uploading a PDF, the system automatically extracts and structures key metadata using the Groq API (`llama-3.3-70b-versatile`):
* Title and Authors
* Research Area
* Main Contributions (Bullet points)
* Methodology and Results Summaries

### 3. Novelty Scoring Engine
Includes a heuristic evaluation module that analyzes the "Contributions" section of a paper. It calculates a normalized novelty score (0.0 - 1.0) based on the density and distinctness of the stated contributions, providing a quick metric for paper uniqueness.

### 4. Comparative Analysis
The system aggregates extracted data from all uploaded papers into a dynamic comparison table. This allows users to review methodologies and results side-by-side without manually toggling between documents.

### 5. Citation-Backed Q&A
The chat interface enforces strict grounded generation. Every answer provided by the assistant includes specific "Source" citations, pointing users to the exact file and text chunk used to generate the response.

## Technical Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain (Chains, Prompts, Document Loaders)
* **Vector Store:** ChromaDB (In-memory)
* **LLM Inference:** Groq API (Llama-3.3-70b)
* **Embeddings:** Ollama (Llama-3.1 local)
* **Keyword Search:** Rank-BM25

## Setup and Usage

1.  **Environment:**
    Ensure you have Python 3.10+ installed.

2.  **Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Local Embeddings:**
    This project uses local embeddings to ensure privacy and speed. Install [Ollama](https://ollama.com/) and pull the model:
    ```bash
    ollama pull llama3.1
    ```

4.  **Configuration:**
    Create a `.env` file in the root directory and add your Groq API key:
    ```bash
    GROQ_API_KEY=your_key_here
    ```

5.  **Run:**
    ```bash
    streamlit run app_research.py
    ```

## Project Structure

* `app_research.py`: Main Streamlit application entry point.
* `src/rag_system.py`: Implementation of the Hybrid Search logic (Vector + BM25).
* `src/paper_analyzer.py`: PDF ingestion, cleaning, and summarization logic.
* `src/chains.py`: LLM prompt construction and QA chain definition.
* `src/evaluation.py`: Novelty scoring heuristic logic.
* `src/config.py`: Centralized configuration using Pydantic settings.
