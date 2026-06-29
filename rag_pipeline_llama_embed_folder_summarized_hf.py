"""
Hugging Face-local RAG ingestion pipeline.

This is the local-transformers variant of
rag_pipeline_llama_embed_folder_summarized.py. It does not call an embedding
HTTP server. Instead, it loads a Hugging Face embedding model locally and uses it
as the LangChain embedding function for Chroma.

Expected input CSV columns:
    pt, t, event, value

Example:
    python rag_pipeline_llama_embed_folder_summarized_hf.py \
        --folder ./tabular \
        --output-folder ./bundle/tabular \
        --infix uid_infix \
        --embedding-model Qwen/Qwen3-Embedding-8B \
        ingest
"""

import csv
import gzip
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer


class HFEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by a local Hugging Face model."""

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
            print(f"Embedding batch {i // self.batch_size + 1} ({len(batch)} texts)")

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


def process_demographic_event(event_name: str, value: str) -> Tuple[str, str]:
    if value.upper() == "NA" and ":" in event_name:
        parts = event_name.split(":")
        if len(parts) == 2:
            prefix, suffix = parts
            return prefix, suffix
    return event_name, value


def summarise_event_series(df: pd.DataFrame) -> Tuple[str, List[int]]:
    df = df.sort_values("t")
    df["t_dt"] = pd.to_datetime(df["t"], unit="s")
    start = df["t_dt"].min()
    end = df["t_dt"].max()

    original_event = df["event"].iloc[0]
    new_event, new_value = process_demographic_event(original_event, df["value"].iloc[0])
    if new_event != original_event:
        df = df.copy()
        df["event"] = new_event
        df["value"] = new_value

    n = len(df)
    text = f"Event '{new_event}' (original: '{original_event}') from {start} to {end} ({n} samples).\n"

    try:
        vals = pd.to_numeric(df["value"], errors="coerce")
        numeric_count = vals.notna().sum()

        if numeric_count / n >= 0.8:
            vals = vals.dropna()
            stats = {
                "mean": vals.mean(),
                "min": vals.min(),
                "max": vals.max(),
                "std": vals.std(),
            }
            first_row = df.iloc[0]
            last_row = df.iloc[-1]
            min_row = df.loc[vals.idxmin()]
            max_row = df.loc[vals.idxmax()]

            text += (
                f"Mean = {stats['mean']:.2f}, "
                f"Min = {stats['min']}, "
                f"Max = {stats['max']}, "
                f"Std = {stats['std']:.2f}. "
                f"First value: {first_row['value']} at {first_row['t_dt']}. "
                f"Last value: {last_row['value']} at {last_row['t_dt']}. "
                f"Min value: {min_row['value']} at {min_row['t_dt']}. "
                f"Max value: {max_row['value']} at {max_row['t_dt']}. "
            )
        else:
            value_counts = df["value"].value_counts()
            text += f"Value distribution: {dict(value_counts.head(5))}. "
            if len(value_counts) > 5:
                text += f"(and {len(value_counts) - 5} more unique values). "
    except Exception:
        value_counts = df["value"].value_counts()
        text += f"Value distribution: {dict(value_counts.head(5))}. "
        if len(value_counts) > 5:
            text += f"(and {len(value_counts) - 5} more unique values). "

    text += f"Notable times: first={start}, last={end}. "

    if n <= 12:
        points = [f" - [{row['t_dt']}]: {row['value']}" for _, row in df.iterrows()]
        text += "All values:\n" + "\n".join(points) + "\n"

    return text.strip(), df["_index"].tolist()


def parse_timestamp(ts_value) -> int:
    if ts_value is None:
        raise ValueError("Timestamp is None")

    try:
        return int(ts_value)
    except ValueError:
        pass

    ts_str = str(ts_value).replace("Z", "+00:00")
    try:
        ts = datetime.fromisoformat(ts_str)
        return int(ts.timestamp())
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot parse timestamp: {ts_value}") from e


def read_csv_to_df(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, mode="rt", newline="") as f:
        header = f.readline().strip().split(",")
        f.seek(0)

        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                row["t"] = parse_timestamp(row["t"])
                row["_index"] = i
                rows.append(row)
            except ValueError:
                continue

    return pd.DataFrame(rows, columns=header + ["_index"])


def batch(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def build_vectorstore_summaries(
    csv_path: Path,
    chroma_dir: Path,
    embedding: Embeddings,
    infix: str,
    chroma_batch_size: int = 512,
) -> None:
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
    if df.empty:
        print(f"Warning: no valid timestamped rows in {csv_path}")
        return

    docs = []
    mapping_data = {}

    for event_name, grp in df.groupby("event"):
        summary, row_indices = summarise_event_series(grp)
        summary_id = str(uuid.uuid4())

        mapping_data[summary_id] = {
            "event": str(event_name),
            "row_indices": row_indices,
            "summary_text": summary,
        }

        docs.append(
            Document(
                page_content=summary,
                metadata={
                    "event": str(event_name),
                    "t_start": int(grp["t"].min()),
                    "t_end": int(grp["t"].max()),
                    "uuid": summary_id,
                },
            )
        )

    chroma_dir.mkdir(parents=True, exist_ok=True)
    mapping_file_path = chroma_dir / "summary_mapping.json"
    with open(mapping_file_path, "w") as f:
        import json
        json.dump(mapping_data, f, indent=2)
    print(f"Saved summary mapping to {mapping_file_path}")

    if chroma_batch_size >= 5461:
        raise ValueError(f"Chroma batch_size max is 5461; got {chroma_batch_size}")

    for chunk in batch(docs, chroma_batch_size):
        print(".", end="", flush=True)
        vectorstore.add_documents(chunk)

    print(f"\nIndexed {len(docs)} event summaries into {chroma_dir}.")

    try:
        del vectorstore
    except Exception as e:
        print(f"Warning: failed to cleanup vectorstore for {chroma_dir}: {e}")


def ingest_folder(
    input_folder: Path,
    output_folder: Path,
    infix: str,
    embedding_model: str,
    device: str,
    embedding_batch_size: int,
    max_length: int,
    chroma_batch_size: int,
    trust_remote_code: bool,
) -> None:
    embeddings_dir = output_folder / infix
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    files_to_ingest = list(input_folder.glob("*.csv.gz"))
    random.shuffle(files_to_ingest)

    print(f"Loading local HF embedding model: {embedding_model}")
    embedding = HFEmbeddings(
        model_name=embedding_model,
        device=device,
        batch_size=embedding_batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
    )

    processed_count = 0
    failed_count = 0

    for csv_path in files_to_ingest:
        print(f"Working on {csv_path}")
        chroma_dir = embeddings_dir / csv_path.stem
        actual_chroma_dir = chroma_dir.with_name(f"{infix}_{chroma_dir.name}")

        if actual_chroma_dir.exists():
            print(f"Skipping {csv_path.stem}: already exists")
            processed_count += 1
            continue

        try:
            build_vectorstore_summaries(
                csv_path,
                chroma_dir,
                embedding=embedding,
                infix=infix,
                chroma_batch_size=chroma_batch_size,
            )
            if actual_chroma_dir.exists():
                processed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"Failed to process {csv_path}: {e}")
            failed_count += 1

    print(f"\nProcessed {processed_count} patients successfully, {failed_count} failed")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["ingest"])
    parser.add_argument("--folder", default=".", help="Folder containing .csv.gz files")
    parser.add_argument("--output-folder", default="N/A", help="Folder to output embedding directories/files")
    parser.add_argument("--infix", default="embedding", help="Sub-folder name for vector DB")
    parser.add_argument("--embedding-model", required=True, help="HF repo ID or local snapshot path")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=int(os.environ.get("EMBEDDING_BATCH_SIZE", 8)))
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--chroma-batch-size", type=int, default=512)
    parser.add_argument("--no-trust-remote-code", action="store_true")

    args = parser.parse_args()

    folder = Path(args.folder)
    output_folder = folder if args.output_folder == "N/A" else Path(args.output_folder)

    if args.action == "ingest":
        ingest_folder(
            input_folder=folder,
            output_folder=output_folder,
            infix=args.infix,
            embedding_model=args.embedding_model,
            device=args.device,
            embedding_batch_size=args.embedding_batch_size,
            max_length=args.max_length,
            chroma_batch_size=args.chroma_batch_size,
            trust_remote_code=not args.no_trust_remote_code,
        )


if __name__ == "__main__":
    main()

