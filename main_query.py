import argparse

from dotenv import load_dotenv

from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME
from src.llm_handler import get_gemini_response
from src.vector_store import load_vector_store


def main():
    """
    Loads the vector store, performs a similarity search, and sends the
    retrieved context to Gemini for final processing.
    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Query the vector database and get a structured answer from Gemini."
    )
    parser.add_argument("query", type=str, help="The query to search for.")
    parser.add_argument(
        "-k", type=int, default=3, help="Number of document chunks to retrieve."
    )
    args = parser.parse_args()

    vector_db = load_vector_store(EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR)
    if not vector_db:
        print(
            "Failed to load vector store. Please build it first using 'main_build_db.py'."
        )
        return

    print(f"Searching for '{args.query}' with k={args.k}...")
    retrieved_docs = vector_db.similarity_search(args.query, k=args.k)

    if not retrieved_docs:
        print("Could not find any relevant documents in the vector store.")
        return

    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    print("\n--- Retrieved Context ---")
    print(context)
    print("\n" + "=" * 50 + "\n")

    gemini_response = get_gemini_response(context, args.query)

    print("--- Formatted Response from Gemini ---")
    print(gemini_response)


if __name__ == "__main__":
    main()
