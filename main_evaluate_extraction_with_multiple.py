import json

import pandas as pd
from dotenv import load_dotenv

from src.config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    GROUND_TRUTH_CSV_PATH,
    RESULTS_CSV_PATH,
    TARGET_VARIABLES,
)
from src.data_loader import load_ground_truth
from src.llm_handler import get_multi_variable_extraction
from src.re_ranker import re_rank_documents
from src.vector_store import load_vector_store


def main():
    load_dotenv()
    print("Starting RAG extraction pipeline with vector retrieval and re-ranking...")

    ground_truth_df = load_ground_truth(GROUND_TRUTH_CSV_PATH)
    vector_db = load_vector_store(EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR)

    all_candidate_docs = []
    print("\nExecuting sequential retrieval for each target variable...")
    for variable in TARGET_VARIABLES:
        print(f"  - Retrieving for: '{variable}'")
        # Perform vector search for each specific variable
        retrieved_docs = vector_db.similarity_search(variable, k=10)
        all_candidate_docs.extend(retrieved_docs)

    # De-duplicate the combined list of candidate documents
    unique_docs_map = {doc.page_content: doc for doc in all_candidate_docs}
    unique_candidates = list(unique_docs_map.values())
    print(f"\nRetrieved {len(unique_candidates)} unique candidate chunks in total.")

    primary_query = (
        f"Financial performance analysis covering {', '.join(TARGET_VARIABLES)}"
    )
    ranked_docs = re_rank_documents(primary_query, unique_candidates)
    print("Re-ranked all candidates for final selection.")

    # Select the top N chunks to form the final context
    # final_docs = ranked_docs[:7]
    context = "\n\n---\n\n".join([doc.page_content for doc in ranked_docs])

    print(f"\n--- Final Context (Built from {len(ranked_docs)} unique chunks) ---")
    print(context)
    print("\n" + "=" * 50 + "\n")

    extracted_data = get_multi_variable_extraction(context, TARGET_VARIABLES)
    print("\n--- Extracted Data (JSON from LLM) ---")
    print(json.dumps(extracted_data, indent=2))

    comparison_results = []
    for variable, extracted_value in extracted_data.items():
        gt_values = ground_truth_df[ground_truth_df["variable_name"] == variable][
            "value"
        ].tolist()
        comparison_results.append(
            {
                "variable_name": variable,
                "ground_truth_values": gt_values,
                "extracted_value": extracted_value,
            }
        )

    results_df = pd.DataFrame(comparison_results)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)

    print(f"\n--- Comparison Results (saved to {RESULTS_CSV_PATH}) ---")
    print(results_df)
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
