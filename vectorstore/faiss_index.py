import faiss
import numpy as np


class FAISSIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, chunks):
        """
        embeddings : numpy array
        chunks : list of chunk dictionaries
        """

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query_embedding, k=5):

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for idx in indices[0]:
            if idx != -1:
                results.append(self.metadata[idx])

        return results