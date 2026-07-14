import pandas as pd

from embeddings.embedder import Embedder
from vectorstore.faiss_index import FAISSIndex
from retrieval.retriever import Retriever
from indexing.build_index import build_index
from benchmark.metrics import recall_at_k
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever

def evaluate_pipeline(
    pdf_path,
    benchmark_path,
    chunk_size=500,
    overlap=100,
    embedding_dim=384,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_method ="faiss"
):

    # ----------------------------
    # Load benchmark CSV
    # ----------------------------
    benchmark = pd.read_csv(benchmark_path)

    # Clean column names (prevents hidden bugs)
    benchmark.columns = benchmark.columns.str.strip()

    # ----------------------------
    # Initialize components
    # ----------------------------
    embedder = Embedder(embedding_model)
    faiss_index = FAISSIndex(dim=embedding_dim)
    if retrieval_method == "faiss":
        retriever = Retriever(embedder, faiss_index)

    elif retrieval_method == "bm25":
    
          retriever = BM25Retriever(faiss_index.chunks)
    elif retrieval_method == "hybrid":
        retrievar = HybridRetriever(embedder, faiss_index)


    else:
    
        raise ValueError("Invalid retrieval method.")

   

    # ----------------------------
    # Build index
    # ----------------------------
    build_index(
        file_path=pdf_path,
        embedder=embedder,
        faiss_index=faiss_index,
        chunk_size=chunk_size,
        overlap=overlap
    )

    print(f"Indexed {faiss_index.index.ntotal} chunks from {pdf_path}")

    # ----------------------------
    # Metrics counters
    # ----------------------------
    total = len(benchmark)

    recall1_total = 0
    recall3_total = 0
    recall5_total = 0

    results = []

    # ----------------------------
    # Evaluation loop
    # ----------------------------
    for _, row in benchmark.iterrows():

        question = row["question"]
        expected_doc = str(row["expected_doc"]).strip()
        expected_page = int(row["expected_page"])

        retrieved = retriever.retrieve(question, k=5)

        # Safety check (avoid crash)
        if not retrieved:
            continue

        r1 = recall_at_k(retrieved, expected_doc, expected_page, 1)
        r3 = recall_at_k(retrieved, expected_doc, expected_page, 3)
        r5 = recall_at_k(retrieved, expected_doc, expected_page, 5)

        recall1_total += r1
        recall3_total += r3
        recall5_total += r5

        top1 = retrieved[0]

        results.append({
            "Question": question,
            "Expected Doc": expected_doc,
            "Expected Page": expected_page,

            "Top1 Doc": top1.get("doc_id", ""),
            "Top1 Page": int(top1.get("page", -1)),
            "Top1 Chunk": top1.get("chunk_id", ""),

            "Recall@1": r1,
            "Recall@3": r3,
            "Recall@5": r5
        })

    # ----------------------------
    # Final metrics
    # ----------------------------
    metrics = {
        "Recall@1": (recall1_total / total) * 100,
        "Recall@3": (recall3_total / total) * 100,
        "Recall@5": (recall5_total / total) * 100
    }

    results_df = pd.DataFrame(results)

    return metrics, results_df