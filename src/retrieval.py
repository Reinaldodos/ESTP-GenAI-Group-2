import faiss


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def retrieve(
    question,
    embedding_model,
    index,
    chunks,
    top_k=3,
):
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0],
    ):
        chunk = chunks[index_number]

        results.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": float(score),
        })

    return results
