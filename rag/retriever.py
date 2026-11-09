import numpy as np


def find_relevant_chunks(
    query,
    chunks,
    embeddings,
    query_embedding,
    top_k=4,
):
    """
    Find the most relevant document chunks
    using cosine similarity.
    """

    if not chunks:
        return []

    if embeddings is None:
        return []

    if len(embeddings) == 0:
        return []

    # Embeddings are normalized,
    # so dot product gives cosine similarity.
    scores = np.dot(
        embeddings,
        query_embedding,
    )

    # Get highest-scoring chunks.
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "chunk": chunks[index],
                "score": float(scores[index]),
            }
        )

    return results