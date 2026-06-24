from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embeddings(self, texts):
        embeddings = self.model.encode(texts)
        return np.array(embeddings, dtype="float32")