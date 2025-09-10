from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql.functions import col, concat_ws, explode, udf
from pyspark.sql.types import ArrayType, StringType


def process_documents(df, chunk_size, chunk_overlap):
    """
    Takes a Spark DataFrame, concatenates text sections, splits them into chunks,
    and returns a new DataFrame with one row per chunk.
    """
    section_cols = [c for c in df.columns if c.startswith("section_")]
    df_full_text = df.withColumn(
        "full_text", concat_ws("\n\n", *[col(c) for c in section_cols])
    )

    def chunk_text_func(text):
        if text is None:
            return []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        return text_splitter.split_text(text)

    chunker_udf = udf(chunk_text_func, ArrayType(StringType()))
    chunked_df = df_full_text.withColumn("chunks", chunker_udf(col("full_text")))
    exploded_df = chunked_df.select(
        "filename", "cik", "year", explode("chunks").alias("chunk")
    )

    return exploded_df
