"""
retrieval.py
============
Semantic similarity search module using ChromaDB and local SentenceTransformers.
"""

from app.ingestion import get_embedding_model, get_chroma_client
from app.config import COLLECTION_NAME


def retrieve_top_k_chunks(query: str, top_k: int = 3):
    """
    Encodes the user query locally and retrieves the top-k most similar document chunks from ChromaDB.
    """
    model = get_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode([query], convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    if results and "ids" in results and results["ids"]:
        ids = results["ids"][0]
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)

        for doc_id, text, meta, dist in zip(ids, docs, metadatas, distances):
            # Cosine similarity in ChromaDB: cosine distance d in [0, 2], similarity = 1 - (d / 2) or 1 - d
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            retrieved.append({
                "id": doc_id,
                "text": text,
                "metadata": meta,
                "distance": dist,
                "similarity": similarity
            })

    return retrieved
