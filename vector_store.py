import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        self.dimension = dimension

        # FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)

        # store original text chunks
        self.texts = []

    def add(self, chunks, embeddings):
        """
        chunks: list of strings
        embeddings: numpy array (float32)
        """

        # 🚨 FORCE correct type (THIS FIXES YOUR ERROR)
        embeddings = np.array(embeddings, dtype="float32")

        # safety check
        if len(embeddings) != len(chunks):
            raise ValueError("Chunks and embeddings size mismatch")

        # add to FAISS
        self.index.add(embeddings)

        # store text for retrieval
        self.texts.extend(chunks)

    def search(self, query_embedding, k=3):
        """
        returns top-k relevant text chunks
        """

        query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])

        return results