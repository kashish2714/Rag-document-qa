from rank_bm25 import BM25Okapi
import re


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

        # Tokenize all chunk texts
        self.tokenized_chunks = [
            self.tokenize(chunk["text"])
            for chunk in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def tokenize(self, text):
        """
        Lowercase + simple word tokenization.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def retrieve(self, query, k=5):
        """
        Returns top-k retrieved chunks.
        """
        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = []

        for idx in ranked_indices:
            results.append(self.chunks[idx])

        return results