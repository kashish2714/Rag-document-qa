from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:

    def __init__(self, embedder, faiss_index):

        self.faiss = Retriever(embedder, faiss_index)
        self.bm25 = BM25Retriever(faiss_index.chunks)

    def retrieve(self, query, k=5):

        faiss_results = self.faiss.retrieve(query, k=10)
        bm25_results = self.bm25.retrieve(query, k=10)

        scores = {}

        # Reciprocal Rank Fusion
        for rank, chunk in enumerate(faiss_results):
            key = (
                chunk["doc_id"],
                chunk["page"],
                chunk["chunk_id"]
            )

            scores.setdefault(key, {"score": 0, "chunk": chunk})
            scores[key]["score"] += 1 / (60 + rank)

        for rank, chunk in enumerate(bm25_results):
            key = (
                chunk["doc_id"],
                chunk["page"],
                chunk["chunk_id"]
            )

            scores.setdefault(key, {"score": 0, "chunk": chunk})
            scores[key]["score"] += 1 / (60 + rank)

        ranked = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [x["chunk"] for x in ranked[:k]]