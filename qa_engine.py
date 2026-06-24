import os
from groq import Groq
from dotenv import load_dotenv

from embeddings import EmbeddingModel

load_dotenv()


class QAEngine:
    def __init__(self, vector_store):
        self.embedder = EmbeddingModel()
        self.vector_store = vector_store

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def ask(self, question):
        query_embedding = self.embedder.get_embeddings([question])[0]

        relevant_chunks = self.vector_store.search(
            query_embedding,
            k=3
        )

        context = "\n\n".join(relevant_chunks)

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an intelligent document question-answering assistant.

Use ONLY the information provided in the context.

Instructions:
- Answer naturally in your own words.
- Do not copy large portions of the document.
- Summarize when appropriate.
- Keep answers concise and informative.
- If the answer is not available in the context, reply:
  'The information is not available in the uploaded document.'
"""
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{question}

Answer:
"""
                }
            ],
            temperature=0.2,
            max_tokens=300
        )

        return response.choices[0].message.content