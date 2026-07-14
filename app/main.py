from fastapi import FastAPI
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from vectorstore.faiss_index import FAISSIndex
from retrieval.retriever import Retriever
from llm.groq_client import GroqClient
from llm.prompt_builder import build_prompt
from indexing.build_index import build_index

app = FastAPI()

# init
embedder = Embedder()
faiss_index = FAISSIndex(dim=384)  # MiniLM dim
retriever = Retriever(embedder, faiss_index)
llm = GroqClient()


@app.post("/ingest")
def ingest(file_path: str):

    build_index(
        file_path,
        embedder,
        faiss_index,
        chunk_size=500,
        overlap=100
    )
    return{
        "message":"PDF indexed successfully"
    }

    pages = load_pdf(file_path)

    all_chunks = []
    for page in pages:
        chunks = chunk_text(
            page,
            chunk_size=chunk_size,
            overlap=overlap
        )

        all_chunks.extend(chunks)
    texts=[chunk["text"] for chunk in all_chunks]
    embeddings = embedder.encode(texts)
    faiss_index.add(embeddings, all_chunks)

    return {"status": "indexed", "chunks": len(all_chunks)}

@app.post("/ask")
def ask(query: str):

    contexts = retriever.retrieve(query, k=5)

    print("\nRetrieved Chunks:")
    for chunk in contexts:
        print(chunk)

    prompt = build_prompt(query, contexts)
    answer = llm.generate(prompt)

    return {
        "query": query,
        "answer": answer,
        "contexts": contexts
    }