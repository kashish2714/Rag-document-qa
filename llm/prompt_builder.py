def build_prompt(query, contexts):
    context_text = "\n\n".join(
        [f"[{i}] {c}" for i, c in enumerate(contexts)]
    )

    prompt = f"""
You are a helpful assistant.

Use ONLY the context below to answer.

Context:
{context_text}

Question:
{query}

If answer is not in context, say "Not found in documents."
"""
    return prompt