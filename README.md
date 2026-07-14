# Evaluated RAG Pipeline for PDF Question Answering

## Overview

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline for document question answering over PDF files. The system combines dense vector retrieval, sparse lexical retrieval, and large language models to generate grounded responses with source attribution.

Unlike a standard RAG application, this project focuses on systematic evaluation of retrieval quality. An evaluation framework was developed to benchmark retrieval performance using Recall@1, Recall@3, and Recall@5 across multiple chunking strategies, overlap settings, embedding models, and retrieval methods.

---

## Features

- Modular PDF ingestion pipeline
- Recursive document chunking
- Sentence-Transformer embeddings
- FAISS vector indexing
- BM25 lexical retrieval
- Hybrid retrieval combining dense and sparse search
- Groq LLM integration for answer generation
- Source-aware responses with page and chunk references
- Retrieval evaluation framework using benchmark question-answer pairs
- Automated experimentation across chunk sizes, overlaps, and embedding models

---

## System Architecture

```
PDF Document
      │
      ▼
PDF Loader
      │
      ▼
Recursive Chunking
      │
      ▼
Sentence Embeddings
      │
      ▼
FAISS Index
      │
      ├───────────────┐
      ▼               │
 Dense Retrieval      │
                      │
BM25 Retrieval        │
      │               │
      └────Hybrid─────┘
            │
            ▼
Prompt Construction
            │
            ▼
Groq LLM
            │
            ▼
Answer + Source Attribution
```

---

## Repository Structure

```
app/
    main.py

benchmark/
    benchmark.csv
    evaluate.py
    experiment.py
    results/

embeddings/
    embedder.py

ingestion/
    loader.py
    chunker.py

llms/
    groq_client.py

retrieval/
    retriever.py
    bm25_retriever.py
    hybrid_retriever.py

vectorstore/
    faiss_store.py

data/
    Telephonic uncanny.pdf
```

---

## Experimental Setup

The retrieval pipeline was evaluated using an annotated benchmark dataset.

### Evaluation Metrics

- Recall@1
- Recall@3
- Recall@5

### Chunk Sizes

- 300
- 500
- 700
- 900
- 1200

### Chunk Overlaps

- 0
- 50
- 100
- 150

### Embedding Models

- sentence-transformers/all-MiniLM-L6-v2
- BAAI/bge-small-en-v1.5
- intfloat/e5-small-v2

### Retrieval Methods

- FAISS
- BM25
- Hybrid Retrieval

---

## Results

The evaluation framework was used to compare multiple retrieval configurations.

Representative results include:

| Configuration | Recall@1 | Recall@3 | Recall@5 |
|--------------|---------:|---------:|---------:|
| Baseline Configuration | 59.77% | 86.21% | 95.40% |
| Best BGE Configuration | 79.31% | 95.40% | 98.85% |
| Best E5 Configuration | 74.71% | 97.70% | 100.00% |

The experiments demonstrate that retrieval performance is significantly influenced by chunk size, overlap strategy, and embedding model selection.

---

## Evaluation Framework

A benchmark dataset consisting of manually curated document questions was created to evaluate retrieval quality.

For each query, the framework records:

- Expected document
- Expected page
- Top retrieved document
- Top retrieved page
- Retrieved chunk
- Recall@1
- Recall@3
- Recall@5

This enables reproducible comparison between different retrieval strategies.

---

## Technologies

- Python
- FastAPI
- FAISS
- Rank-BM25
- Sentence Transformers
- Groq API
- Pandas
- NumPy

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
cd RAG-PROJECT
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file.

```
GROQ_API_KEY=your_api_key
```

Start the API.

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000/docs
```

---

## Future Improvements

- Cross-encoder reranking
- Query expansion
- Metadata-aware retrieval
- Multi-document retrieval
- Retrieval precision metrics (MRR, nDCG)
- LLM answer evaluation
- Streaming responses