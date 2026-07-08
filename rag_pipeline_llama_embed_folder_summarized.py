"""
End-to-end LangGraph RAG pipeline using Chroma + LlamaCppEmbeddings
for any compressed CSV files with columns:
    pt   : int
    t    : int  (Unix epoch seconds)
    event: str
    value: str

Usage:
    python src/main.py ingest --folder ./folder --host localhost --port 8080 --infix my_run
    python src/main.py query --folder ./folder --host localhost --port 8080 --infix my_run "..."
"""

import gzip
import csv
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Generator, Tuple

import random
import httpx
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
#from langchain.schema import Document
from langchain_core.embeddings import Embeddings 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import time
import numpy as np


# ------------------------------------------------------------------
# 1.  LangGraph state & nodes
# ------------------------------------------------------------------
class State(BaseModel):
    question: str
    context: List[Document] = Field(default_factory=list)
    distances: List[float] = Field(default_factory=list)
    answer: str = ""

class LlamaCppEmbeddings(Embeddings):
    def __init__(self, base_url: str = "http://localhost:8080", server_url: str = "http://localhost:8080/embedding", max_retries: int = 3, retry_backoff: float = 2.0, model_path: str = ""):
        self.base_url = base_url.rstrip("/")
        self.server_url = server_url
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.model_path = model_path

    def embed_documents(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        # Use environment variable if batch_size not explicitly provided
        if batch_size is None:
            import os
            batch_size = int(os.environ.get('EMBEDDING_BATCH_SIZE', 16))

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            attempt = 0
            while attempt < self.max_retries:
                try:
                    print(f"Embedding batch {i//batch_size + 1}, attempt {attempt+1}")
                    r = httpx.post(
                        # f"{self.base_url}/embedding",
                        # f"{self.base_url}/v1/embeddings",
                        self.server_url,
                        json={"input": batch, "model": self.model_path},
                        timeout=120
                    )
                    r.raise_for_status()
                    response_data = r.json()
                    # Handle multiple response formats:
                    # 1. sglang format: {"data": [{"embedding": [...], "index": 0}, ...]}
                    # 2. llama.cpp native: {"embedding": [[...]]} (nested array)
                    # 3. OpenAI-compatible list: [{"embedding": [[...]]}]
                    if isinstance(response_data, dict) and "data" in response_data:
                        # sglang format
                        for item in response_data["data"]:
                            if "embedding" in item:
                                emb = item["embedding"]
                                # Handle if embedding is nested [[...]] vs [...]
                                if isinstance(emb[0], list):
                                    results.append(emb[0])
                                else:
                                    results.append(emb)
                    elif isinstance(response_data, list):
                        # List format
                        for item in response_data:
                            if "embedding" in item:
                                emb = item["embedding"]
                                # Handle if embedding is nested [[...]] vs [...]
                                if isinstance(emb[0], list):
                                    results.append(emb[0])
                                else:
                                    results.append(emb)
                    elif isinstance(response_data, dict) and "embedding" in response_data:
                        # Single dict format
                        emb = response_data["embedding"]
                        # Handle if embedding is nested [[...]] vs [...]
                        if isinstance(emb[0], list):
                            results.append(emb[0])
                        else:
                            results.append(emb)
                    else:
                        raise ValueError(f"Unexpected response format: {response_data}")
                    break
                except (httpx.TimeoutException, httpx.HTTPError) as e:
                    print(f"Embedding request failed: {e}")
                    attempt += 1
                    if attempt < self.max_retries:
                        sleep_time = self.retry_backoff ** attempt
                        print(f"Retrying in {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
                    else:
                        raise RuntimeError(f"Failed to embed batch after {self.max_retries} attempts") from e
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def transform_event_name(event_name: str, base_url: str = "http://localhost:8080", transform: bool = False) -> str:
    """
    Transform the event name into a concise meaningful phrase without loss of information
    using the LLM server. Only transforms if transform flag is True.
    """
    # If transform flag is False, return original event name
    if not transform:
        return event_name
    
    prompt = f"""
    Paraphrase into a concise, meaningful phrase without losing any information.
    Increase clinical readability while preserving all original meaning.
    Event name: {event_name}
    """
    
    attempt = 0
    max_retries = 3
    retry_backoff = 2.0
    
    print(prompt)

    while attempt < max_retries:
        try:
            r = httpx.post(
                f"{base_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 128,
                    "temperature": 0.0,
                    "stop": ["\n"]
                },
                timeout=60
            )
            r.raise_for_status()
            response_data = r.json()
            transformed = response_data.get("content", "").strip()
            
            if transformed:
                return transformed
            else:
                raise ValueError("Empty response from LLM")
                
        except (httpx.TimeoutException, httpx.HTTPError, ValueError) as e:
            print(f"Event transformation request failed: {e}")
            attempt += 1
            if attempt < max_retries:
                sleep_time = retry_backoff ** attempt
                print(f"Retrying in {sleep_time:.1f}s...")
                # time.sleep(sleep_time)
            else:
                print(f"Failed to transform event after {max_retries} attempts, using original")
                return event_name

def process_demographic_event(event_name: str, value: str) -> Tuple[str, str]:
    """
    Process demographic events with format "...:<suffix_part>" and value NA.
    Returns a tuple of (transformed_event, new_value).
    """
    # Check if this is a demographic event with NA value
    if value.upper() == "NA" and ":" in event_name:
        parts = event_name.split(":")
        if len(parts) == 2:
            prefix, suffix = parts
            return prefix, suffix
    
    # Return original values if not a demographic event with NA
    return event_name, value

# ------------------------------------------------------------------
# 2.  Summarise per event type
# ------------------------------------------------------------------
def summarise_event_series(df: pd.DataFrame, base_url: str = "http://localhost:8080", transform_event_name_flag: bool = False) -> Tuple[str, List[int]]:
    """
    Given a DataFrame for ONE event type, return a human-readable summary
    that always includes:
    - start / end timestamps
    - count & frequency
    - descriptive stats (mean, min, max, std)
    - values at argmin, argmax, first, last times
    - if ≤12 rows, additionally list every individual point
    
    Returns:
        Tuple of (summary_text, list_of_original_row_indices)
    """
    # Convert t back to datetime
    df = df.sort_values("t")
    df["t_dt"] = pd.to_datetime(df["t"], unit="s")
    start = df["t_dt"].min()
    end = df["t_dt"].max()
    
    # Get the original event name
    original_event = df['event'].iloc[0]
    
    # Transform the event name using LLM (only if transform_event_name is True)
    transformed_event = transform_event_name(original_event, base_url, transform_event_name_flag)
    
    # Process demographic events (always do this)
    new_event, new_value = process_demographic_event(original_event, df['value'].iloc[0])
    
    # If it's a demographic event with NA, update the dataframe
    if new_event != original_event:
        df = df.copy()
        df['event'] = new_event
        df['value'] = new_value
    
    n = len(df)
    text = f"Event '{transformed_event}' (original: '{original_event}') from {start} to {end} ({n} samples).\n"
    
    # Attempt to convert values to numeric
    try:
        vals = pd.to_numeric(df["value"], errors="coerce")
        numeric_count = vals.notna().sum()
        
        # If 80% or more values are numeric, treat as numeric serie2s
        if numeric_count / n >= 0.8:
            vals = vals.dropna()
            stats = {
                "mean": vals.mean(),
                "min": vals.min(),
                "max": vals.max(),
                "std": vals.std(),
            }
            
            # Add numeric statistics
            text += (
                f"Mean = {stats['mean']:.2f}, "
                f"Min = {stats['min']}, "
                f"Max = {stats['max']}, "
                f"Std = {stats['std']:.2f}. "
            )
            
            # Get values at specific time points
            first_row = df.iloc[0]
            last_row = df.iloc[-1]
            
            # Find min and max value rows
            min_idx = vals.idxmin()
            max_idx = vals.idxmax()
            min_row = df.loc[min_idx]
            max_row = df.loc[max_idx]
            
            # Format the values with their timestamps
            text += (
                f"First value: {first_row['value']} at {first_row['t_dt']}. "
                f"Last value: {last_row['value']} at {last_row['t_dt']}. "
                f"Min value: {min_row['value']} at {min_row['t_dt']}. "
                f"Max value: {max_row['value']} at {max_row['t_dt']}. "
            )
            
            # Skip value distribution for numeric data
        else:
            # Treat as categorical if <80% numeric
            value_counts = df["value"].value_counts()
            text += f"Value distribution: {dict(value_counts.head(5))}. "
            if len(value_counts) > 5:
                text += f"(and {len(value_counts) - 5} more unique values)"
                
    except Exception:
        # Fallback to categorical if conversion fails
        value_counts = df["value"].value_counts()
        text += f"Value distribution: {dict(value_counts.head(5))}. "
        if len(value_counts) > 5:
            text += f"(and {len(value_counts) - 5} more unique values)"
    
    # Always add notable times
    text += f"Notable times: first={start}, last={end}. "
    
    # List all values if ≤12
    if n <= 12:
        points = []
        for _, row in df.iterrows():
            points.append(f" - [{row['t_dt']}]: {row['value']}")
        text += "All values:\n" + "\n".join(points) + "\n"
    
    # Get the original row indices
    row_indices = df["_index"].tolist()
    
    return text.strip(), row_indices

def parse_timestamp(ts_value) -> int:
    """Parse timestamp from either Unix epoch int or ISO 8601 string."""
    if ts_value is None:
        raise ValueError("Timestamp is None")
    # Try Unix epoch first
    try:
        return int(ts_value)
    except ValueError:
        pass
    # Try ISO 8601 format
    ts_str = str(ts_value).replace("Z", "+00:00")
    try:
        ts = datetime.fromisoformat(ts_str)
        return int(ts.timestamp())
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot parse timestamp: {ts_value}") from e

def read_csv_to_df(path: Path) -> pd.DataFrame:
    """Read gzipped CSV -> DataFrame with numeric t.
       Drop rows whose 't' is NA or cannot be converted to int or ISO 8601.
    """
    rows = []
    with gzip.open(path, mode="rt", newline="") as f:
        # Read the header to determine the columns
        header = f.readline().strip().split(",")
        f.seek(0)  # Reset file position
        
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                row["t"] = parse_timestamp(row["t"])
                # Add an index field to track original row position
                row["_index"] = i
                rows.append(row)
            except ValueError:
                # skip rows with invalid / missing timestamp
                continue
    
    # Use the columns from the header
    return pd.DataFrame(rows, columns=header + ["_index"])

def build_vectorstore_summaries(csv_path: Path,
                                chroma_dir: Path,
                                batch_size: int = 512,
                                embedding: Embeddings = None,
                                base_url: str = "http://localhost:8080",
                                infix: str = "embedding",
                                transform_event_name_flag: bool = False) -> None:
    chroma_dir = chroma_dir.with_name(f"{infix}_{chroma_dir.name}")
    if chroma_dir.exists():
        print(f"Skipping; existing vector DB at {chroma_dir}")
        return

    vectorstore = Chroma(
        collection_name="labs",
        embedding_function=embedding,
        persist_directory=str(chroma_dir),
        collection_metadata={"hnsw:space": "l2", "hnsw:num_threads": 1},
    )

    df = read_csv_to_df(csv_path)
    docs = []

    # Create a mapping file to store summary-to-row-indices relationships
    mapping_file_path = chroma_dir / "summary_mapping.json"
    mapping_data = {}

    # group by event type
    for event_name, grp in df.groupby("event"):
        summary, row_indices = summarise_event_series(grp, base_url, transform_event_name_flag)
        
        # Generate a unique ID for this summary
        summary_id = str(uuid.uuid4())
        
        # Store the mapping
        mapping_data[summary_id] = {
            "event": str(event_name),
            "row_indices": row_indices,
            "summary_text": summary
        }
        
        docs.append(
            Document(
                page_content=summary,
                metadata={
                    "event": str(event_name),
                    "t_start": int(grp["t"].min()),
                    "t_end":   int(grp["t"].max()),
                    "uuid":    summary_id,
                    # REMOVED: "row_indices": row_indices,  # This causes the error
                },
            )
        )

    # Save the mapping file
    import json
    with open(mapping_file_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print(f"Saved summary mapping to {mapping_file_path}")

    if batch_size >= 5461:
        print(f"Chroma batch_size max is 5461 --- user set it to {batch_size}")
        raise Exception

    for chunk in batch(docs, batch_size):
        print('.', end='')
        vectorstore.add_documents(chunk)
    print(f"\nIndexed {len(docs)} event summaries into {chroma_dir}.")

    # Explicit cleanup to release HNSW threads
    try:
        del vectorstore
    except Exception as e:
        print(f"Warning: Failed to cleanup vectorstore for {chroma_dir}: {e}")
        print("         HNSW threads may not be properly released, which could lead to thread accumulation.")

def batch(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def ingest_folder(input_folder: Path,
                  output_folder: Path,
                  batch_size: int = 512,
                  base_url: str = "http://localhost:8080",
                  server_url: str = "http://localhost:8080/embedding",
                  infix: str = "embedding",
                  transform_event_name_flag: bool = False,
                  model_path: str = "") -> None:
    import multiprocessing as mp
    import time

    # Use 'spawn' context for better compatibility
    ctx = mp.get_context('spawn')

    embeddings_dir = output_folder / infix
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    files_to_ingest = list(input_folder.glob("*.csv.gz"))
    random.shuffle(files_to_ingest)

    print(f"Processing {len(files_to_ingest)} patients using multiprocessing...")

    processed_count = 0
    failed_count = 0

    for csv_path in files_to_ingest:
        print(f'Working on {csv_path}')
        chroma_dir = embeddings_dir / csv_path.stem

        # Account for the name transformation in build_vectorstore_summaries
        actual_chroma_dir = chroma_dir.with_name(f"{infix}_{chroma_dir.name}")

        # Skip if already exists
        if actual_chroma_dir.exists():
            print(f"Skipping {csv_path.stem}: already exists")
            processed_count += 1
            continue

        # Process in subprocess
        process = ctx.Process(
            target=_process_single_patient,
            args=(csv_path, chroma_dir, batch_size, base_url, server_url,
                  infix, transform_event_name_flag, model_path)
        )

        process.start()
        process.join(timeout=300)  # 5 minute timeout

        # Handle timeout
        if process.exitcode is None:
            print(f"Warning: Process for {csv_path.stem} timed out")
            try:
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            except:
                pass
            failed_count += 1
        elif process.exitcode == 0:
            # Check the ACTUAL chroma directory that was created
            if actual_chroma_dir.exists():
                print(f"Successfully processed {csv_path.stem}")
                processed_count += 1
            else:
                print(f"Warning: No output created for {csv_path.stem}")
                failed_count += 1
        else:
            print(f"Process for {csv_path.stem} failed with exit code {process.exitcode}")
            failed_count += 1

        del process
        time.sleep(0.05)

    print(f"\nProcessed {processed_count} patients successfully, {failed_count} failed")


def _process_single_patient(csv_path, chroma_dir, batch_size, base_url, server_url, infix, transform_event_name_flag, model_path):
    """Process a single patient in a subprocess."""
    embedding_object = LlamaCppEmbeddings(base_url, server_url, model_path=model_path)
    build_vectorstore_summaries(csv_path, chroma_dir, batch_size=batch_size,
                                embedding=embedding_object, base_url=base_url,
                                infix=infix, transform_event_name_flag=transform_event_name_flag)
    # Process exits naturally

# ------------------------------------------------------------------
# 3.  LangGraph pipeline nodes (unchanged)
# ------------------------------------------------------------------
def retrieve(state: State, chroma_dir: Path, k: int = 20, base_url: str = "http://localhost:8080", server_url: str = "http://localhost:8080/embedding", model_path: str = "") -> State:
    embedding = LlamaCppEmbeddings(base_url, server_url, model_path=model_path)
    vectorstore = Chroma(
        collection_name="labs",
        embedding_function=embedding,
        persist_directory=str(chroma_dir),
        collection_metadata={"hnsw:space": "l2", "hnsw:num_threads": 1},
    )
    
    # Get the query embedding
    query_embedding = embedding.embed_query(state.question)
    
    # Search with scores to get cosine distances
    results_with_scores = vectorstore.similarity_search_with_score(
        state.question, 
        k=k
    )
    
    # Extract documents and scores
    docs = []
    distances = []
    for doc, score in results_with_scores:
        docs.append(doc)
        # Convert score to cosine distance (lower is more similar)
        # Chroma returns L2 distance by default, but we can treat it as a similarity measure
        distances.append(score)
        
    # Sort documents by distance (ascending - lower distance means more similar)
    sorted_docs = [doc for _, doc in sorted(zip(distances, docs), key=lambda x: x[0])]
    
    # Store both the documents and their distances in the state
    state.context = sorted_docs
    state.distances = sorted(distances)
    
    return state

def generate(state: State, base_url: str = "http://localhost:8080") -> State:
    llm = ChatOpenAI(
        openai_api_base=f"{base_url}/v1",
        openai_api_key="dummy",
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=512,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful clinical assistant. "
                   "Answer the question using only the lab data provided below."),
        ("user", "Context:\n{context}\n\nQuestion:\n{question}")
    ])
    context_str = "\n".join([d.page_content for d in state.context])
    # print(f'{len(prompt[0])} tokens')
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context_str,
        "question": state.question
    })
    state.answer = answer
    return state

# ------------------------------------------------------------------
# 4.  CLI interface
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["ingest", "query"])
    parser.add_argument("--folder", default=".", help="Folder containing .csv.gz files")
    parser.add_argument("--output-folder", default="N/A", help="Folder to output embedding directories/files")
    parser.add_argument("--host", default="localhost", help="Host for LLM server")
    parser.add_argument("--port", type=int, default=8080, help="Port for LLM server")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding requests")
    parser.add_argument("--infix", default="embedding", help="Sub-folder name for vector DB")
    parser.add_argument("--endpoint-type", default="completion", help="endpoint: llama.cpp's /embedding (use 'completion') or oai compatible /v1/embeddings (use 'openai')")
    parser.add_argument("--transform-event-name-flag", action="store_true", help="Transform event names using LLM")
    parser.add_argument("--model-path", default="", help="Full path to embedding model for sglang server")
    parser.add_argument("question", nargs="?", default="")
    args = parser.parse_args()

    folder = Path(args.folder)
    if args.output_folder == "N/A":
        output_folder = folder
    else:
        output_folder = Path(args.output_folder)
    base_url = f"http://{args.host}:{args.port}"
    if args.endpoint_type == "completion":
        server_url = f'{base_url}/embedding' 
    else:
        server_url = f'{base_url}/v1/embeddings' 

    if args.action == "ingest":
        ingest_folder(folder, output_folder, batch_size=args.batch_size,
                      base_url=base_url, server_url=server_url, infix=args.infix,
                      transform_event_name_flag=args.transform_event_name_flag,
                      model_path=args.model_path)
        sys.exit(0)

    if args.action == "query":
        if not args.question:
            print("Provide a question after 'query'")
            sys.exit(1)

        embeddings_dir = output_folder / args.infix
        if not embeddings_dir.exists():
            print("Warning: output-folder not found, trying input-folder for embeddings")
            embeddings_dir = folder / args.infix
            if not embeddings_dir.exists():
                print("No embeddings directory found. Run `ingest` first.")
                sys.exit(1)

        available = sorted(embeddings_dir.iterdir())
        # import pdb
        # pdb.set_trace()
        if not available:
            print("No embeddings found. Run `ingest` first.")
            sys.exit(1)

        chroma_dir = available[2]  # pick first available

        def _run_retrieve(state: State) -> State:
            return retrieve(state, chroma_dir, base_url=base_url, server_url=server_url, model_path=args.model_path)

        def _run_generate(state: State) -> State:
            return generate(state, base_url=base_url)

        workflow = StateGraph(State)
        workflow.add_node("retrieve", _run_retrieve)
        workflow.add_node("generate", _run_generate)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        app = workflow.compile()

        result = app.invoke({"question": args.question})
        print("--- Retrieved context ---")
        for i, doc in enumerate(result["context"]):
            distance = result["distances"][i]
            print(f"- [Distance: {distance:.4f}] {doc.page_content}")
        print("\n--- Llama-server answer ---")
        print(result["answer"])
