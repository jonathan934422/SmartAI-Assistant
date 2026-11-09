from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


# Load the embedding model once.
model = SentenceTransformer(MODEL_NAME)


def create_embeddings(texts):
    """Convert text into numerical embeddings."""

    if not texts:
        return []

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings