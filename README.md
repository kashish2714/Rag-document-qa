# RAG Document QA System

An AI-powered document question-answering system built using Retrieval-Augmented Generation (RAG).

## Features

- Upload PDF and TXT documents
- Automatic document parsing
- Text chunking with overlap
- Semantic embeddings using Sentence Transformers
- FAISS vector database for retrieval
- Context-aware answers using Groq LLM
- FastAPI backend
- Chat-style frontend interface

## Tech Stack

- Python
- FastAPI
- FAISS
- Sentence Transformers
- Groq API
- HTML
- CSS
- JavaScript

## How It Works

1. Upload a document
2. Extract text from the document
3. Split text into chunks
4. Generate embeddings
5. Store embeddings in FAISS
6. Retrieve relevant chunks for user questions
7. Generate answers using Groq LLM

## Future Improvements

- Source citations
- Multi-document support
- Conversation memory
- Better UI/UX
- Deployment to cloud

