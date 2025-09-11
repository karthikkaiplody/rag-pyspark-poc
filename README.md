# PySpark and LangChain RAG Pipeline with Gemini

This project is a proof-of-concept for building a Retrieval-Augmented Generation (RAG) pipeline. It uses PySpark for data processing and LangChain for chunking and ChromaDB vector storage, and then queries Google's Gemini Pro model to extract structured information from the retrieved text.

## Data Configuration

This project is configured to read data from a Parquet file. Before running, please update the data source settings in `src/config.py`:

-   `PARQUET_FILE_PATH`: Set this to the location of your Parquet file (e.g., `"./data/my_data.parquet"`).
-   `CIK_FILTER`: Set this to the specific CIK you want to process (e.g., `"5272"`). If you want to process the **entire Parquet file**, set this value to `None`.

## Environment Variables

Before running the project, create a `.env` file in the root directory to store your API keys and configuration variables. For example:

```
GEMINI_API_KEY=your_gemini_api_key_here
JAVA_HOME=your_java_home
```

## How to Run

### 1. Build the Vector Database

This script will start a Spark session with the specified configurations, load the Parquet data, process it, create embeddings, and save the vector database to the `vector_db/` directory.

```bash
python main_build_db.py
```
This only needs to be done once, or whenever the source data changes.

### 2. Query the Database and Get a Structured Answer

Once the database is built, we can run queries against it using the `main_query.py` script.

**Usage:**
```bash
python main_query.py "Query text here"
```

### 3. Query the Database and Get a Structured Answer

Once the database is built, to run the full pipeline use the below command.

**Usage:**
```bash
python main_evaluate_extraction_with_multiple.py
```


**Examples:**
```bash
python main_query.py "What was the pre-tax income and were there any catastrophe losses?"

python main_query.py "information on product lines and income" -k 2
```

**Future Enhancements::**
 - [ ] Add script to compare retrieved values against ground truth dataset.
 - [ ] Add Hybrid search approach to add more recall and retrieve more relavent chunks.
 - [ ] Experiment with the Chunk-Size and Chunk Overlap to see if the performance increases.
 - [ ] Implement and experiment other chunking strategies like Semanting Chunking, Markdown Chunking, LLM Based Chunking
       
