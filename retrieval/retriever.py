class Retriever:
    def __init__(self, embedder, faiss_index):
        self.embedder = embedder
        self.index = faiss_index

    def retrieve(self, query, k=5):
        q_emb = self.embedder.encode([query])[0]
        results = self.index.search(q_emb, k)
        return results