import pandas as pd
from pyspark.sql import DataFrame, SparkSession


def get_spark_session(app_name="Retrival Pipeline"):
    """
    Initializes and returns a Spark session with specific configurations.
    """
    return (
        SparkSession.builder.config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        .config("spark.ui.port", "4040")
        .appName(app_name)
        .getOrCreate()
    )


def load_parquet_data(
    spark: SparkSession, file_path: str, cik_to_filter: str = None
) -> DataFrame:
    """
    Loads data from a Parquet file and optionally filters it by a CIK.
    """
    print(f"Loading data from Parquet file: {file_path}")
    spark_df = (
        spark.read.option("spark.sql.files.maxPartitionBytes", "100m")
        .option("spark.sql.shuffle.partitions", "10")
        .parquet(file_path)
    )
    if cik_to_filter:
        print(f"Filtering data for CIK: {cik_to_filter}")
        return spark_df.filter(spark_df.cik == cik_to_filter)
    return spark_df


def load_ground_truth(file_path: str) -> pd.DataFrame:
    """Loads the ground truth data from a CSV file into a pandas DataFrame."""
    print(f"Loading ground truth from: {file_path}")
    return pd.read_csv(file_path)
