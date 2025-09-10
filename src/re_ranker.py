from sentence_transformers import CrossEncoder

# Initialize the re-ranker model once and reuse it.
# This model is specifically trained for ranking relevance.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def re_rank_documents(query: str, documents: list) -> list:
    """
    Re-ranks a list of documents based on their relevance to a query using a CrossEncoder.

    Args:
        query (str): The search query.
        documents (list): A list of document strings to be re-ranked.

    Returns:
        list: The sorted list of documents, most relevant first.
    """
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs]
