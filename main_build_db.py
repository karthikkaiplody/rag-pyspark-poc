from dotenv import load_dotenv

from src.config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CIK_FILTER,
    EMBEDDING_MODEL_NAME,
    PARQUET_FILE_PATH,
    PROCESSED_CHUNKS_PATH,
)
from src.data_loader import get_spark_session, load_parquet_data
from src.processing import process_documents
from src.vector_store import build_vector_store


def main():
    load_dotenv()

    spark = get_spark_session()
    df = load_parquet_data(spark, PARQUET_FILE_PATH, CIK_FILTER)

    if df.rdd.isEmpty():
        print("DataFrame is empty after loading and filtering. Exiting.")
        spark.stop()
        return

    print(f"Loaded and filtered {df.count()} documents.")

    # Chunking for vector-db
    df_chunks = process_documents(df, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {df_chunks.count()} text chunks.")
    df_chunks.show(5, truncate=False)

    build_vector_store(df_chunks, EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR)

    spark.stop()
    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
