def chunk_text(page, chunk_size=500, overlap=100):
    """
    Splits a page into overlapping chunks while preserving metadata.

    Args:
        page (dict): {
            "doc_id": "...",
            "page": 1,
            "text": "..."
        }

    Returns:
        List[dict]: List of chunk dictionaries.
    """

    text = page["text"]

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append({
            "id": f'{page["doc_id"]}_p{page["page"]}_c{chunk_id}',
            "doc_id": page["doc_id"],
            "page": page["page"],
            "chunk_id": chunk_id,
            "text": text[start:end]
        })

        chunk_id += 1
        start = end - overlap

    return chunks