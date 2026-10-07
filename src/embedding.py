from sentence_transformers import SentenceTransformer


def create_embedding_model(model_name):
    return SentenceTransformer(model_name)


def create_embeddings(
    model,
    chunks,
    normalize=True,
):
    texts = [chunk["text"] for chunk in chunks]

    return model.encode(
        texts,
        normalize_embeddings=normalize,
    )