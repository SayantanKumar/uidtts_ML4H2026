"""
Hugging Face-local comparison and ranking for anchor queries and RAG summaries.

This is the local-transformers variant of anchor_match_ranker.py. It does not
call embedding or reranker HTTP servers. It loads the embedding model and
reranker model directly with Hugging Face Transformers.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import random
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from langchain_chroma import Chroma
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

VERBOSE = False

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api="blas")
except ImportError:
    threadpool_limits = None


def log_threads(label: str = ""):
    if not VERBOSE:
        return
    count = threading.active_count()
    print(f"[THREADS] {label}: {count} Python threads")


def load_anchor_queries(anchor_file: Path, hide_tags: bool = False) -> Dict[str, List[str]]:
    with open(anchor_file, "r") as f:
        data = json.load(f)

    if hide_tags:
        tag_pattern = re.compile(r"<[^>]+>\s*")
        cleaned = {}
        for event, queries in data.items():
            cleaned[event] = [tag_pattern.sub("", q).strip() for q in queries] if queries else queries
        return cleaned

    return data


class HFEmbeddings:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 8192,
        trust_remote_code: bool = True,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden_states[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.inference_mode():
                outputs = self.model(**encoded)
                pooled = self._last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)

            results.extend(pooled.float().cpu().tolist())

        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class HFReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 4,
        max_length: int = 8192,
        trust_remote_code: bool = True,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        self.token_false_id = self._single_token_id("no")
        self.token_true_id = self._single_token_id("yes")

        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def _single_token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if not encoded:
                raise ValueError(f"Could not encode reranker token: {token}")
            token_id = encoded[0]
        return int(token_id)

    @staticmethod
    def format_instruction(instruction: str, query: str, doc: str) -> str:
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _prepare_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        for i, token_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + token_ids + self.suffix_tokens

        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return {key: value.to(self.device) for key, value in inputs.items()}

    @torch.inference_mode()
    def _score_pairs(self, pairs: List[str]) -> List[float]:
        scores: List[float] = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            inputs = self._prepare_inputs(batch)
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores.extend(batch_scores[:, 1].exp().float().cpu().tolist())

        return scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        instruction: str = None,
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        pairs = [self.format_instruction(instruction, query, doc) for doc in documents]
        scores = self._score_pairs(pairs)

        rows = [
            {"score": float(score), "index": idx, "document": documents[idx]}
            for idx, score in enumerate(scores)
        ]
        rows.sort(key=lambda row: row["score"], reverse=True)

        if top_n is not None:
            rows = rows[:top_n]

        return rows


_EMBEDDING_MODEL = None


def get_embedding_model(
    embedding_model: str,
    device: str,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
) -> HFEmbeddings:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print(f"Loading local HF embedding model: {embedding_model}")
        _EMBEDDING_MODEL = HFEmbeddings(
            model_name=embedding_model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )
    return _EMBEDDING_MODEL


def load_rag_summaries(rag_dir: Path) -> List[Dict[str, Any]]:
    log_threads(f"load_rag_summaries START for {rag_dir.name}")

    chroma_dirs = []
    for item in rag_dir.iterdir():
        if item.is_dir() and (item / "chroma.sqlite3").exists():
            chroma_dirs.append(item)
        elif item.name == "chroma.sqlite3":
            chroma_dirs.append(rag_dir)

    if not chroma_dirs:
        chroma_dirs = [rag_dir]

    chroma_dir = chroma_dirs[0]
    vectorstore = Chroma(
        collection_name="labs",
        persist_directory=str(chroma_dir),
        collection_metadata={"hnsw:space": "l2", "hnsw:num_threads": 1},
    )

    results = vectorstore.get(include=["metadatas", "documents", "embeddings"])

    summaries = []
    for i, doc in enumerate(results["documents"]):
        if i < len(results["embeddings"]) and results["embeddings"][i] is not None:
            summaries.append({
                "content": doc,
                "metadata": results["metadatas"][i],
                "embedding": results["embeddings"][i],
                "id": results["ids"][i],
            })
        else:
            print(f"Warning: no embedding found for document {i}")

    try:
        del vectorstore
    except Exception:
        pass

    print(f"Loaded {len(summaries)} summaries with embeddings")
    return summaries if summaries else None


def compare_anchor_vs_summaries(
    anchor_query: str,
    summaries: List[Dict[str, Any]],
    embedding: HFEmbeddings,
    k: int = 5,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    anchor_embedding = np.array(embedding.embed_query(anchor_query)).reshape(1, -1)

    if not summaries:
        print("Warning: no summaries with embeddings found")
        return [], []

    summary_embeddings = np.array([s["embedding"] for s in summaries])

    if threadpool_limits is not None:
        with threadpool_limits(limits=1, user_api="blas"):
            similarities = cosine_similarity(anchor_embedding, summary_embeddings)[0]
    else:
        similarities = cosine_similarity(anchor_embedding, summary_embeddings)[0]

    sorted_indices = np.argsort(similarities)[::-1]
    top_k_indices = sorted_indices[:k]

    return [summaries[i] for i in top_k_indices], [float(similarities[i]) for i in top_k_indices]


def cross_encoder_reranker(
    anchor_query: str,
    candidate_summaries: List[Dict[str, Any]],
    reranker: HFReranker,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    task = "Given the description of putative tabular data matches in the query, retrieve relevant tabular summaries, if any."
    documents = [summary["content"] for summary in candidate_summaries]
    results = reranker.rerank(
        query=anchor_query,
        documents=documents,
        instruction=task,
        top_n=len(documents),
    )

    reranked_summaries = []
    reranked_scores = []
    for result in results:
        idx = result["index"]
        reranked_summaries.append(candidate_summaries[idx])
        reranked_scores.append(float(result["score"]))

    return reranked_summaries, reranked_scores


def process_case_report(
    case_name: str,
    anchor_file: Path,
    rag_dir: Path,
    embedding: HFEmbeddings,
    reranker: HFReranker = None,
    k: int = 10,
    hide_tags: bool = False,
) -> Dict[str, Any]:
    anchor_queries = load_anchor_queries(anchor_file, hide_tags=hide_tags)
    summaries = load_rag_summaries(rag_dir)
    if not summaries:
        print(f"Skipping {case_name}: failed to load embeddings")
        return None

    results = {}
    query_count = 0

    for event, queries in anchor_queries.items():
        event_results = []
        if queries is None:
            continue

        for query in queries:
            query_count += 1
            contextual_query = "Information regarding: " + event + ", in particular: " + query

            candidate_summaries, candidate_scores = compare_anchor_vs_summaries(
                contextual_query,
                summaries,
                embedding=embedding,
                k=k,
            )

            if reranker is not None and candidate_summaries:
                reranked_summaries, reranked_scores = cross_encoder_reranker(
                    contextual_query,
                    candidate_summaries,
                    reranker,
                )
                final_summaries = reranked_summaries
                final_scores = reranked_scores
            else:
                final_summaries = candidate_summaries
                final_scores = candidate_scores

            best_summary = final_summaries[0]["content"] if final_summaries else ""
            best_score = float(final_scores[0]) if final_scores else 0.0

            event_results.append({
                "anchor_query": contextual_query,
                "initial_candidates": [
                    {"content": s["content"], "score": float(score)}
                    for s, score in zip(candidate_summaries, candidate_scores)
                ],
                "reranked_results": [
                    {"content": s["content"], "score": float(score)}
                    for s, score in zip(final_summaries, final_scores)
                ],
                "best_summary": best_summary,
                "best_score": best_score,
            })

        results[event] = event_results

    print(f"Processed {case_name}: {query_count} anchor queries")
    return results


def find_rag_case_dir(case_name: str, rag_dir: Path) -> Path:
    numeric_part = case_name.split("-")[0]

    for item in rag_dir.iterdir():
        if item.is_dir() and numeric_part in item.name:
            return item

    for item in rag_dir.iterdir():
        if item.is_dir():
            chroma_files = list(item.glob("**/chroma.sqlite3"))
            if chroma_files:
                return chroma_files[0].parent

    return None


def process_folder(
    anchor_dir: Path,
    rag_dir: Path,
    embedding_model: str,
    reranker_model: str,
    device: str,
    embedding_batch_size: int,
    reranker_batch_size: int,
    max_length: int,
    k: int,
    use_cross_encoder: bool,
    overwrite: bool,
    hide_tags: bool,
    trust_remote_code: bool,
) -> None:
    results_dir = anchor_dir / "comparison_results"
    results_dir.mkdir(exist_ok=True)

    embedding = get_embedding_model(
        embedding_model=embedding_model,
        device=device,
        batch_size=embedding_batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
    )

    reranker = None
    if use_cross_encoder:
        print(f"Loading local HF reranker model: {reranker_model}")
        reranker = HFReranker(
            model_name=reranker_model,
            device=device,
            batch_size=reranker_batch_size,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )

    json_files = list(anchor_dir.glob("*.json"))
    random.shuffle(json_files)

    processed_count = 0
    failed_count = 0

    for anchor_file in json_files:
        case_name = anchor_file.stem
        while Path(case_name).suffix:
            case_name = Path(case_name).stem

        output_file = results_dir / f"{case_name}_results.json"
        if output_file.exists() and not overwrite:
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                if data:
                    print(f"Skipping {case_name}: results already exist")
                    continue
            except (json.JSONDecodeError, IOError):
                print(f"File {output_file} exists but is empty/corrupted, reprocessing")

        rag_case_dir = find_rag_case_dir(case_name, rag_dir)
        if not rag_case_dir:
            print(f"Warning: no RAG directory found for {case_name}")
            failed_count += 1
            continue

        try:
            results = process_case_report(
                case_name=case_name,
                anchor_file=anchor_file,
                rag_dir=rag_case_dir,
                embedding=embedding,
                reranker=reranker,
                k=k,
                hide_tags=hide_tags,
            )

            if results:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {output_file}")
                processed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"Failed to process {case_name}: {e}")
            failed_count += 1

    print(f"\nProcessed {processed_count} cases successfully, {failed_count} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["compare"])
    parser.add_argument("--folder", default=".", help="Folder containing case data")
    parser.add_argument("--anchor-subdir", default="anchors", help="Directory containing anchor queries")
    parser.add_argument("--rag-subdir", default="embedding", help="Directory containing RAG embeddings")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embedding-model", required=True, help="HF repo ID or local snapshot path")
    parser.add_argument("--reranker-model", required=True, help="HF repo ID or local snapshot path")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=int(os.environ.get("EMBEDDING_BATCH_SIZE", 8)))
    parser.add_argument("--reranker-batch-size", type=int, default=int(os.environ.get("RERANKER_BATCH_SIZE", 4)))
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--no-cross-encoder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hide-tags", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Ignored compatibility arguments from the server-based script.
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reranker-host", default="localhost")
    parser.add_argument("--reranker-port", type=int, default=8082)
    parser.add_argument("--endpoint-type", default="openai")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--concurrency", type=int, default=1)

    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    folder = Path(args.folder)
    anchor_dir = folder / args.anchor_subdir
    rag_dir = folder / args.rag_subdir

    if not anchor_dir.exists():
        raise FileNotFoundError(f"Anchor subdirectory not found: {anchor_dir}")
    if not rag_dir.exists():
        raise FileNotFoundError(f"RAG directory not found: {rag_dir}")

    if args.action == "compare":
        process_folder(
            anchor_dir=anchor_dir,
            rag_dir=rag_dir,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
            device=args.device,
            embedding_batch_size=args.embedding_batch_size,
            reranker_batch_size=args.reranker_batch_size,
            max_length=args.max_length,
            k=args.k,
            use_cross_encoder=not args.no_cross_encoder,
            overwrite=args.overwrite,
            hide_tags=args.hide_tags,
            trust_remote_code=not args.no_trust_remote_code,
        )
        print("Comparison and ranking completed for all case reports.")


if __name__ == "__main__":
    main()

