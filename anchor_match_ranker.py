"""
Comparison and ranking system for anchor queries and RAG summaries.
Compares anchor query outputs with RAG summaries, surfaces best k for each anchor,
and uses a SentenceTransformer cross-encoder re-ranker to identify the best K overall per case report.

Usage:
    python src/comparison_ranker.py compare --folder ./folder --anchor-dir anchors --rag-dir embedding --host localhost --port 8080 --k 10

Example:
    python comparison_ranker.py compare --folder ./data --anchor-dir my_anchors --rag-dir my_embedding --host localhost --port 8080 --k 5
"""

# LIMIT THREADS BEFORE IMPORTING THREAD-SPAWNING LIBRARIES
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit OpenBLAS threads
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # Limit Accelerate threads
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Limit NumExpr threads

import torch
torch.set_num_threads(1)  # Limit PyTorch intra-op threads
torch.set_num_interop_threads(1)  # Limit PyTorch inter-op threads

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse

import httpx
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import threading

# Global verbose flag for thread logging
VERBOSE = False

# ACTIVELY LIMIT BLAS THREAD POOLS
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api='blas')
    print("[SETUP] BLAS thread pools limited to 1 thread")
except ImportError:
    print("[WARNING] threadpoolctl not available - install with: pip install threadpoolctl")

# Debug function to track thread creation
def log_threads(label: str = ""):
    """Log current thread count for debugging (only if VERBOSE is True)."""
    if not VERBOSE:
        return
    count = threading.active_count()
    if label:
        print(f"[THREADS] {label}: {count} Python threads")
    else:
        print(f"[THREADS] Current: {count} Python threads")

    # Also log native thread count from OS
    try:
        import subprocess
        pid = os.getpid()
        result = subprocess.run(['ps', '-o', 'nlwp=', '-p', str(pid)], capture_output=True, text=True)
        native_threads = result.stdout.strip()
        print(f"[THREADS] {label}: {native_threads} OS threads (from ps)")
    except:
        pass

    return count

def load_anchor_queries(anchor_file: Path, hide_tags: bool = False) -> Dict[str, List[str]]:
    """Load anchor queries from JSON file, optionally stripping tags."""
    with open(anchor_file, 'r') as f:
        data = json.load(f)
    
    if hide_tags:
        tag_pattern = re.compile(r'<[^>]+>\s*')
        cleaned = {}
        for event, queries in data.items():
            cleaned[event] = [tag_pattern.sub('', q).strip() for q in queries]
        return cleaned
    return data


class LlamaCppEmbeddings:
    def __init__(self, base_url: str = "http://localhost:8080", server_url: str = "http://localhost:8080/embedding", model_path: str = ""):
        self.base_url = base_url.rstrip("/")
        self.server_url = server_url
        self.model_path = model_path

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for text using LLM server."""
        try:
            r = httpx.post(
                # f"{self.base_url}/embedding",
                self.server_url,
                json={"input": [text], "model": self.model_path},
                timeout=30
            )
            # import pdb
            # pdb.set_trace()
            r.raise_for_status()
            response_data = r.json()
            # Handle multiple response formats:
            # 1. sglang format: {"data": [{"embedding": [...], "index": 0}, ...]}
            # 2. llama.cpp native: {"embedding": [[...]]} (nested array)
            # 3. OpenAI-compatible list: [{"embedding": [[...]]}]
            if isinstance(response_data, dict) and "data" in response_data:
                # sglang format
                emb = response_data["data"][0]["embedding"]
            elif isinstance(response_data, list):
                # List format
                emb = response_data[0]["embedding"]
            elif isinstance(response_data, dict) and "embedding" in response_data:
                # Single dict format
                emb = response_data["embedding"]
            else:
                raise ValueError(f"Unexpected response format: {response_data}")

            # Handle if embedding is nested [[...]] vs [...]
            if isinstance(emb[0], list):
                return emb[0]
            else:
                return emb
        except Exception as e:
            print(f"Embedding request failed: {e}")
            return [0.0] * 4096  # Fallback


def load_rag_summaries(rag_dir: Path, base_url: str = "http://localhost:8080", server_url: str = "http://localhost:8080/embedding", model_path: str = "") -> List[Dict[str, Any]]:
    """Load all RAG summaries from Chroma vectorstore using the correct embedding function."""
    log_threads(f"load_rag_summaries START for {rag_dir.name}")
    from langchain_chroma import Chroma

    # Use the same embedding function that was used during ingestion
    embedding = LlamaCppEmbeddings(base_url, server_url, model_path=model_path)

    # Find the actual Chroma database directory
    chroma_dirs = []
    for item in rag_dir.iterdir():
        if item.is_dir():
            # Check if this is a Chroma directory (contains chroma.sqlite3)
            if (item / "chroma.sqlite3").exists():
                chroma_dirs.append(item)
            # Also check for chroma.sqlite3 in the root
        elif item.name == "chroma.sqlite3":
            chroma_dirs.append(rag_dir)

    if not chroma_dirs:
        # If no chroma directories found, try using the rag_dir itself
        chroma_dirs = [rag_dir]

    # Use the first found Chroma directory
    chroma_dir = chroma_dirs[0]

    log_threads("Before Chroma creation")

    # Load the vectorstore
    vectorstore = Chroma(
        collection_name="labs",
        embedding_function=embedding,
        persist_directory=str(chroma_dir),
        collection_metadata={"hnsw:space": "l2", "hnsw:num_threads": 1},
    )
    log_threads("After Chroma creation")

    # Get all documents with metadata AND embeddings
    results = vectorstore.get(include=["metadatas", "documents", "embeddings"])

    # Debug: print what we found
    print(f"Found {len(results['documents'])} documents")
    print(f"Found {len(results['embeddings'])} embeddings")

    summaries = []
    for i, doc in enumerate(results["documents"]):
        # Make sure we have an embedding for this document
        if i < len(results["embeddings"]) and results["embeddings"][i] is not None:
            summaries.append({
                "content": doc,
                "metadata": results["metadatas"][i],
                "embedding": results["embeddings"][i],
                "id": results["ids"][i]
            })
        else:
            print(f"Warning: No embedding found for document {i}")

    print(f"Loaded {len(summaries)} summaries with embeddings")
    if len(summaries)==0:
        print(f"Setting summaries to None")
        summaries = None

    # Explicit cleanup
    try:
        # Access the underlying chromadb client and try to clear it
        if hasattr(vectorstore, '_client'):
            try:
                # For persistent client, we just need to delete the reference
                # The HNSW threads should be cleaned up when the C++ object is destroyed
                pass
            except:
                pass

        del vectorstore

        # Force garbage collection
        import gc
        gc.collect()

        log_threads("After Chroma cleanup")

    except Exception as e:
        print(f"Warning: Failed to cleanup vectorstore for {chroma_dir}: {e}")
        print("         HNSW threads may not be properly released, which could lead to thread accumulation.")

    return summaries

def get_embedding(text: str, base_url: str = "http://localhost:8080", server_url: str = "http://localhost:8080/embedding", model_path: str = "") -> List[float]:
    """Get embedding for text using LLM server."""
    embedding_obj = LlamaCppEmbeddings(base_url, server_url, model_path=model_path)
    return embedding_obj.embed_query(text)


class RerankerClient:
    """Client for sglang reranker server."""
    def __init__(self, host: str = "localhost", port: int = 8082):
        self.base_url = f"http://{host}:{port}"

    def rerank(self, query: str, documents: List[str], instruction: str = None, top_n: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents against query using sglang reranker server.

        Args:
            query: The query text
            documents: List of document texts to rank
            instruction: Optional instruction for the reranker
            top_n: Maximum number of documents to return

        Returns:
            List of dicts with 'score', 'index', and optionally 'document'
        """
        payload = {
            "model": "default",
            "query": query,
            "documents": documents,
            "return_documents": True
        }

        if instruction:
            payload["instruct"] = instruction

        if top_n is not None:
            payload["top_n"] = top_n

        try:
            response = httpx.post(
                f"{self.base_url}/v1/rerank",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Reranker request failed: {e}")
            # Return default scores on failure
            return [{"score": 0.0, "index": i, "document": doc}
                    for i, doc in enumerate(documents)]

def compare_anchor_vs_summaries(anchor_query: str, summaries: List[Dict[str, Any]],
                               base_url: str, server_url: str, model_path: str = "", k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Compare anchor query against all summaries using cosine similarity.
    Return top k summaries with their similarity scores.
    """
    # Get embedding for anchor query
    anchor_embedding = get_embedding(anchor_query, base_url, server_url, model_path)
    anchor_embedding = np.array(anchor_embedding).reshape(1, -1)

    # Check if we have any summaries with embeddings
    if not summaries:
        print("Warning: No summaries with embeddings found")
        return [], []

    # Use pre-computed embeddings from summaries instead of recalculating
    summary_embeddings = np.array([s["embedding"] for s in summaries])

    # LIMIT THREADS FOR COSINE SIMILARITY
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api='blas'):
            # Calculate cosine similarities
            similarities = cosine_similarity(anchor_embedding, summary_embeddings)[0]
    except:
        # Fallback if threadpoolctl not available
        similarities = cosine_similarity(anchor_embedding, summary_embeddings)[0]
    
    # Check if we have any summaries with embeddings
    if not summaries:
        print("Warning: No summaries with embeddings found")
        return [], []
    
    # Use pre-computed embeddings from summaries instead of recalculating
    summary_embeddings = np.array([s["embedding"] for s in summaries])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(anchor_embedding, summary_embeddings)[0]
    
    # Sort by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]
    top_k_indices = sorted_indices[:k]
    
    top_summaries = [summaries[i] for i in top_k_indices]
    top_scores = [similarities[i] for i in top_k_indices]
    
    return top_summaries, top_scores

def format_instruction(instruction: str, query: str, doc: str) -> str:
    """Format the instruction for the reranker model."""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def process_inputs(pairs: List[str], tokenizer: AutoTokenizer, max_length: int, 
                  prefix_tokens: List[int], suffix_tokens: List[int], device) -> Dict[str, Any]:
    """Process input pairs for the reranker model."""
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation='longest_first',
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    
    return inputs


@torch.no_grad()
def compute_logits(inputs: Dict[str, Any], model: AutoModelForCausalLM, 
                  token_true_id: int, token_false_id: int) -> List[float]:
    """Compute logits for the reranker model."""
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


def cross_encoder_reranker(anchor_query: str, candidate_summaries: List[Dict[str, Any]],
                          reranker_client: RerankerClient) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Use Qwen3 reranker via server to re-rank candidate summaries for an anchor query.
    Returns re-ranked summaries and their scores.
    """
    task = 'Given the description of putative tabular data matches in the query, retrieve relevant tabular summaries, if any.'

    # Extract document contents
    documents = [summary["content"] for summary in candidate_summaries]

    # Call reranker server
    results = reranker_client.rerank(
        query=anchor_query,
        documents=documents,
        instruction=task,
        top_n=len(documents)
    )

    # Extract scores and sort by score (descending)
    # Results are already sorted by score from the server
    reranked_summaries = []
    reranked_scores = []

    for result in results:
        idx = result["index"]
        reranked_summaries.append(candidate_summaries[idx])
        reranked_scores.append(result["score"])

    return reranked_summaries, reranked_scores


def process_case_report(case_name: str, anchor_file: Path, rag_dir: Path,
                       base_url: str = "http://localhost:8080",
                       server_url: str = "http://localhost:8080/embedding",
                       model_path: str = "",
                       k: int = 10,
                       use_cross_encoder: bool = True,
                       reranker_client: RerankerClient = None,
                       hide_tags: bool = False) -> Dict[str, Any]:
    """
    Process a single case report: compare anchors vs summaries and optionally re-rank.
    """
    log_threads(f"process_case_report START for {case_name}")

    # Load anchor queries
    anchor_queries = load_anchor_queries(anchor_file, hide_tags=hide_tags)

    # Load RAG summaries with their pre-computed embeddings
    summaries = load_rag_summaries(rag_dir, base_url, server_url, model_path)
    if not summaries:
        print(f'Skipping {case_name}: failed to load embeddings')
        return None

    results = {}
    query_count = 0

    for event, queries in anchor_queries.items():
        event_results = []

        if queries is None:
            continue
        for query in queries:
            query_count += 1

            # Log every 10 queries to track thread growth
            if query_count % 10 == 0:
                log_threads(f"After {query_count} queries")

            contextual_query = 'Information regarding: ' + event + ', in particular: ' + query
            # First stage: get top k summaries using cosine similarity
            candidate_summaries, candidate_scores = compare_anchor_vs_summaries(
                contextual_query, summaries, base_url, server_url, model_path, k
            )

            # Second stage: optional reranker re-ranking
            if use_cross_encoder and reranker_client is not None:
                reranked_summaries, reranked_scores = cross_encoder_reranker(
                    contextual_query, candidate_summaries, reranker_client
                )
                best_summary = reranked_summaries[0]["content"] if reranked_summaries else ""
                best_score = float(reranked_scores[0]) if reranked_scores else 0
                final_results = [
                    {"content": s["content"], "score": float(score)}
                    for s, score in zip(reranked_summaries, reranked_scores)
                ]
            else:
                reranked_summaries = candidate_summaries
                reranked_scores = candidate_scores
                best_summary = candidate_summaries[0]["content"] if candidate_summaries else ""
                best_score = float(candidate_scores[0]) if candidate_scores else 0
                final_results = [
                    {"content": s["content"], "score": float(score)}
                    for s, score in zip(candidate_summaries, candidate_scores)
                ]

            event_results.append({
                "anchor_query": contextual_query,
                "initial_candidates": [
                    {"content": s["content"], "score": float(score)}
                    for s, score in zip(candidate_summaries, candidate_scores)
                ],
                "reranked_results": final_results,
                "best_summary": best_summary,
                "best_score": best_score
            })

        results[event] = event_results

    log_threads(f"process_case_report END for {case_name} (processed {query_count} queries)")
    return results

def _process_single_case_wrapper(args):
    """Wrapper function for multiprocessing - unpacks args and calls process_case_report."""
    (case_name, anchor_file, rag_case_dir, base_url, server_url, model_path,
     k, use_cross_encoder, reranker_host, reranker_port, hide_tags, output_file) = args

    # Create reranker client inside the subprocess
    reranker_client = None
    if use_cross_encoder:
        try:
            reranker_client = RerankerClient(reranker_host, reranker_port)
        except:
            pass

    # Process the case
    results = process_case_report(case_name, anchor_file, rag_case_dir, base_url, server_url, model_path, k,
                                use_cross_encoder, reranker_client, hide_tags)

    if results:
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
        return True
    else:
        print(f'Warning: no results from {case_name}.')
        return False


def subject_id_candidates(case_name: str) -> List[str]:
    """Return likely subject IDs embedded in a case filename."""
    candidates = []

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    add(case_name)
    for token in re.findall(r"(?<!\d)\d{5,}(?!\d)", case_name):
        add(token)

    return candidates


def rag_dir_matches_subject(item_name: str, subject_id: str) -> bool:
    """Match subject IDs as numeric tokens, not arbitrary substrings."""
    return re.search(rf"(^|[^0-9]){re.escape(subject_id)}([^0-9]|$)", item_name) is not None


def find_rag_case_dir(case_name: str, rag_dir: Path) -> Path:
    subject_ids = subject_id_candidates(case_name)

    for item in rag_dir.iterdir():
        if item.is_dir() and any(rag_dir_matches_subject(item.name, subject_id) for subject_id in subject_ids):
            chroma_files = list(item.glob("**/chroma.sqlite3"))
            return chroma_files[0].parent if chroma_files else item

    indexed_dirs = []
    for item in rag_dir.iterdir():
        if item.is_dir():
            chroma_files = list(item.glob("**/chroma.sqlite3"))
            if chroma_files:
                indexed_dirs.append(chroma_files[0].parent)

    if len(indexed_dirs) == 1:
        return indexed_dirs[0]

    return None


def process_folder(anchor_dir: Path, rag_dir: Path,
                  base_url: str = "http://localhost:8080",
                  server_url: str = "http://localhost:8080/embedding",
                  model_path: str = "",
                  k: int = 10,
                  use_cross_encoder: bool = True,
                  reranker_host: str = "localhost",
                  reranker_port: int = 8082,
                  overwrite: bool = False,
                  hide_tags: bool = False,
                  concurrency: int = 1) -> None:
    """
    Process all case reports in folder.
    Only process cases that don't have existing results unless overwrite is True.
    Uses multiprocessing to isolate each patient in its own process for proper thread cleanup.
    """
    import multiprocessing as mp
    import time

    # Use 'spawn' context for better compatibility with torch/transformers
    ctx = mp.get_context('spawn')

    results_dir = anchor_dir/ "comparison_results"
    results_dir.mkdir(exist_ok=True)

    log_threads("process_folder START")

    # Test reranker connection once at start
    if use_cross_encoder:
        print(f"Connecting to reranker server at {reranker_host}:{reranker_port}...")
        try:
            reranker_client = RerankerClient(reranker_host, reranker_port)
            test_result = reranker_client.rerank("test query", ["test doc"])
            print("Reranker server connected successfully.")
            del reranker_client
        except Exception as e:
            print(f"Failed to connect to reranker server: {e}")
            use_cross_encoder = False

    json_files = list(anchor_dir.glob("*.json"))
    random.shuffle(json_files)

    processed_count = 0
    failed_count = 0

    # Build list of cases to process
    cases_to_process = []
    for anchor_file in json_files:
        case_name = anchor_file.stem
        while Path(case_name).suffix:
            case_name = Path(case_name).stem
        output_file = results_dir / f"{case_name}_results.json"

        # Skip if file exists and overwrite is False
        if output_file.exists() and not overwrite:
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                if data:
                    if VERBOSE:
                        print(f"Skipping {case_name}: results already exist")
                    continue
            except (json.JSONDecodeError, IOError):
                print(f"File {output_file} exists but is empty/corrupted, reprocessing...")

        # Find RAG directory
        rag_case_dir = find_rag_case_dir(case_name, rag_dir)
        if rag_case_dir and VERBOSE:
            print(f"Found matching RAG directory: {rag_case_dir}")

        if not rag_case_dir:
            print(f"Warning: No RAG directory found for {case_name}")
            failed_count += 1
            continue

        cases_to_process.append((case_name, anchor_file, rag_case_dir, output_file))

    print(f"Processing {len(cases_to_process)} cases sequentially (to avoid pool hanging issues)")

    for case_name, anchor_file, rag_case_dir, output_file in cases_to_process:
        # Process in subprocess with timeout
        process = ctx.Process(
            target=_process_single_case_process,
            args=(case_name, anchor_file, rag_case_dir, base_url, server_url, model_path,
                  k, use_cross_encoder, reranker_host, reranker_port, hide_tags, output_file)
        )

        process.start()
        process.join(timeout=300)  # 5 minute timeout per case

        # Handle timeout
        if process.exitcode is None:
            print(f"Warning: Process for {case_name} timed out")
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
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    if data:
                        print(f"Successfully processed {case_name}")
                        processed_count += 1
                    else:
                        print(f"Warning: Empty output file for {case_name}")
                        failed_count += 1
                except:
                    print(f"Warning: Could not read output file for {case_name}")
                    failed_count += 1
            else:
                print(f"Warning: No output file created for {case_name}")
                failed_count += 1
        else:
            print(f"Process for {case_name} failed with exit code {process.exitcode}")
            failed_count += 1

        del process
        time.sleep(0.05)

    print(f"\nProcessed {processed_count} cases successfully, {failed_count} failed")
    log_threads("process_folder END")


def _process_single_case_process(case_name, anchor_file, rag_case_dir, base_url, server_url, model_path,
                                  k, use_cross_encoder, reranker_host, reranker_port, hide_tags, output_file):
    """Process a single case in a subprocess. Returns True if successful. Results are written to output_file."""
    # Setup reranker client
    reranker_client = None
    if use_cross_encoder:
        try:
            reranker_client = RerankerClient(reranker_host, reranker_port)
        except:
            pass

    # Process the case
    results = process_case_report(case_name, anchor_file, rag_case_dir, base_url, server_url, model_path, k,
                                use_cross_encoder, reranker_client, hide_tags)

    # Write results to file
    if results:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["compare"])
    parser.add_argument("--folder", default=".", help="Folder containing case data")
    parser.add_argument("--anchor-subdir", default="anchors", help="Directory containing anchor queries (relative)")
    parser.add_argument("--rag-subdir", default="embedding", help="Directory containing RAG embeddings (relative)")
    parser.add_argument("--host", default="localhost", help="Host for embedding server")
    parser.add_argument("--port", type=int, default=8080, help="Port for embedding server")
    parser.add_argument("--reranker-host", default="localhost", help="Host for reranker server")
    parser.add_argument("--reranker-port", type=int, default=8082, help="Port for reranker server")
    parser.add_argument("--k", type=int, default=10, help="Number of top candidates to consider")
    parser.add_argument("--endpoint-type", default="completion", help="endpoint: llama.cpp's /embedding (use 'completion') or oai compatible /v1/embeddings (use 'openai')")
    parser.add_argument("--model-path", default="", help="Full path to embedding model for sglang server")
    parser.add_argument("--no-cross-encoder", action="store_true",
                       help="Disable cross-encoder re-ranking")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any saved files at output locations")
    parser.add_argument("--hide-tags", action="store_true", help="Remove XML/HTML tags from anchor queries before processing")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of cases to process in parallel")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging including thread counts")

    args = parser.parse_args()

    # Set global verbose flag
    global VERBOSE
    VERBOSE = args.verbose

    folder = Path(args.folder)
    anchor_dir = folder / args.anchor_subdir
    rag_dir = folder / args.rag_subdir
    base_url = f"http://{args.host}:{args.port}"
    if args.endpoint_type == "completion":
        server_url = f'{base_url}/embedding' 
    else:
        server_url = f'{base_url}/v1/embeddings' 
    
    if args.action == "compare":
        if not anchor_dir.exists():
            print(f"Anchor subdirectory not found: {anchor_dir}")
            return
        
        if not rag_dir.exists():
            print(f"RAG directory not found: {rag_dir}")
            return
        
        process_folder(anchor_dir, rag_dir, base_url, server_url, args.model_path, args.k,
                      not args.no_cross_encoder, args.reranker_host, args.reranker_port, args.overwrite, args.hide_tags, args.concurrency)
        print("Comparison and ranking completed for all case reports.")

if __name__ == "__main__":
    main()
