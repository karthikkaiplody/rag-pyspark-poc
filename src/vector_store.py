from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def build_vector_store(df_chunks, embedding_model_name, persist_directory):
    """
    Embeds document chunks and builds a persistent Chroma vector store.
    """
    pdf_chunks = df_chunks.toPandas()
    pdf_chunks = pdf_chunks[pdf_chunks["chunk"].str.len() > 0]

    docs_to_embed = pdf_chunks["chunk"].tolist()
    metadata_to_store = pdf_chunks[["filename", "cik", "year"]].to_dict("records")

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    vector_db = Chroma.from_texts(
        texts=docs_to_embed,
        embedding=embedding_model,
        metadatas=metadata_to_store,
        persist_directory=persist_directory,
    )
    print(f"Vector store created and saved to '{persist_directory}'.")
    return vector_db


def load_vector_store(embedding_model_name, persist_directory):
    """
    Loads an existing Chroma vector store from disk.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )
    return vector_db
