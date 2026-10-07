def chunk_text(text, chunk_size=500, overlap=100):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def create_chunks(documents, chunk_size=500, overlap=100):
    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["text"],
            chunk_size,
            overlap,
        )

        for chunk_id, chunk in enumerate(document_chunks):
            chunks.append({
                "source": document["source"],
                "chunk_id": chunk_id,
                "text": chunk,
            })

    return chunks