"""
Anchor query generator for discharge summary event temporal localization.
Generates possible anchor queries based on EHR database events to help locate
discharge summary events in time.

Usage:
    python src/anchor_query_generator.py generate --folder ./folder --host localhost --port 8080 --infix my_run"

Example:
    python anchor_query_generator.py generate --folder ./data --host localhost --port 8080 --infix my_anchors

Recommend:
    Use a SOTA reasoning LLM
"""

import gzip
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages.ai import AIMessage
from langchain_core.exceptions import OutputParserException
from openai import LengthFinishReasonError
import argparse
import sys
import random
import re

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
MENTION_TABLE_RE = re.compile(r"\n\s*Mention table:\s*\n", re.IGNORECASE)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def is_placeholder_anchor_json(data: Any) -> bool:
    """Return True for temporary placeholder JSON written before generation."""
    return isinstance(data, dict) and len(data) == 1 and data.get(" ") is None

def read_prompt_file(file_path: Path) -> str:
    """Read a prompt file (plain text or .gz)."""
    if not file_path.exists():
        return ""
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return f.read()
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

def strip_code_fences(text: str) -> str:
    text = text.strip()
    m = FENCE_RE.search(text)
    return m.group(1).strip() if m else text

def parse_or_repair_json(result: str, llm, attempt: int):
    text = strip_code_fences(result)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try simple repairs first
        try:
            # Try adding closing brace
            return json.loads(text + "}")
        except:
            pass

        # Try closing array and object if truncated mid-array
        try:
            # Remove trailing comma if present, then close arrays/objects
            cleaned = text.rstrip()
            if cleaned.endswith(','):
                cleaned = cleaned[:-1]
            # Count unclosed brackets
            open_braces = cleaned.count('{') - cleaned.count('}')
            open_brackets = cleaned.count('[') - cleaned.count(']')
            # Add necessary closing characters
            closing = ']' * open_brackets + '}' * open_braces
            return json.loads(cleaned + closing)
        except:
            pass

    print(f"JSON parsing llm try {attempt+1}")

    repair_grounding = ChatPromptTemplate.from_messages([
        ("system", "You are a data engineer."),
        ("user", "Fix this so it is valid JSON. Return ONLY the repaired JSON, no explanations.\n\n{bad_json}"),
    ])

    # repair_chain = repair_grounding | llm | StrOutputParser()
    repair_chain = repair_grounding | llm
    repair_result = strip_code_fences(repair_chain.invoke({"bad_json": text}).content)

    return json.loads(repair_result)

def split_report_and_mention_table(case_report_text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split a bundled report into report body and uid4/mention rows."""
    parts = MENTION_TABLE_RE.split(case_report_text, maxsplit=1)
    report_body = parts[0]
    table_text = parts[1] if len(parts) > 1 else ""

    rows = []
    for raw_line in table_text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("uid4"):
            continue
        if "|" not in line:
            continue
        uid, mention = line.split("|", 1)
        uid = uid.strip().strip("<>")
        mention = mention.strip()
        if uid and mention:
            rows.append((uid, mention))

    return report_body, rows

def batched(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, size)
    return [items[i:i + size] for i in range(0, len(items), size)]

def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_uid_contexts(report_body: str,
                         uids: List[str],
                         context_window_chars: int,
                         max_report_chars: int) -> str:
    """Return compact report excerpts around the UID tags for this batch."""
    snippets = []
    used_ranges = []

    for uid in uids:
        pattern = re.compile(rf"<{re.escape(uid)}>(.*?)</{re.escape(uid)}>", re.DOTALL)
        for match in pattern.finditer(report_body):
            start = max(0, match.start() - context_window_chars)
            end = min(len(report_body), match.end() + context_window_chars)

            if any(abs(start - old_start) < 100 and abs(end - old_end) < 100 for old_start, old_end in used_ranges):
                continue

            used_ranges.append((start, end))
            snippets.append(f"[<{uid}>] {clean_space(report_body[start:end])}")

    excerpt = "\n".join(snippets)
    if not excerpt:
        excerpt = clean_space(report_body[:max_report_chars])

    return excerpt[:max_report_chars]

def build_chunk_case_report(report_body: str,
                            rows: List[Tuple[str, str]],
                            context_window_chars: int,
                            max_report_chars: int) -> str:
    uids = [uid for uid, _ in rows]
    excerpts = extract_uid_contexts(report_body, uids, context_window_chars, max_report_chars)
    table_rows = "\n".join(f"{uid}|{mention}" for uid, mention in rows)
    return (
        "Case report excerpts for the UID mentions in this batch:\n"
        f"{excerpts}\n\n"
        "Mention table for this batch only:\n"
        "uid4 | Mention\n"
        f"{table_rows}\n"
    )

def normalize_anchor_queries(anchor_queries: Any,
                             max_queries_per_uid: int) -> Dict[str, List[str]]:
    """Normalize model output into dict[str, list[str]] and cap list size."""
    if not isinstance(anchor_queries, dict):
        return {}

    normalized = {}
    for key, values in anchor_queries.items():
        if key is None:
            continue
        key_text = str(key).strip()
        if not key_text:
            continue

        if isinstance(values, str):
            candidate_values = [values]
        elif isinstance(values, list):
            candidate_values = values
        else:
            continue

        clean_values = []
        seen = set()
        for value in candidate_values:
            value_text = clean_space(str(value))
            if not value_text or value_text in seen:
                continue
            clean_values.append(value_text)
            seen.add(value_text)
            if len(clean_values) >= max_queries_per_uid:
                break

        if clean_values:
            normalized[key_text] = clean_values

    return normalized

def key_contains_uid(key: str, uid: str) -> bool:
    return f"<{uid}>" in key or key.startswith(uid) or f" {uid} " in f" {key} "

def default_anchor_queries(uid: str, mention: str, max_queries_per_uid: int) -> Dict[str, List[str]]:
    """Create conservative fallback queries when the LLM omits a UID."""
    queries = [
        f"Timestamped structured EHR rows that directly document or match: {mention}",
        f"Orders, results, administrations, procedures, imaging, vitals, or nursing chart events associated with: {mention}",
        f"Nearby timestamped events in the admission record that support the timing of: {mention}",
    ]
    return {f"<{uid}> {mention}": queries[:max_queries_per_uid]}

def merge_anchor_queries(target: Dict[str, List[str]],
                         source: Dict[str, List[str]],
                         max_queries_per_uid: int) -> None:
    for key, values in source.items():
        existing = target.setdefault(key, [])
        for value in values:
            if value not in existing:
                existing.append(value)
            if len(existing) >= max_queries_per_uid:
                break

def generate_anchor_queries_batched(case_report_text: str,
                                    base_url: str = "http://localhost:8080",
                                    max_retries: int = 3,
                                    system_prompt_path: Path = Path("templates/aqg_system.txt"),
                                    user_prompt_path: Path = Path("templates/aqg_user.txt"),
                                    model_name: str = 'gpt-4o',
                                    chunk_size: int = 25,
                                    context_window_chars: int = 700,
                                    max_report_chars: int = 25000,
                                    max_queries_per_uid: int = 3) -> Dict[str, List[str]]:
    """
    Generate anchor queries in UID batches for large reports, then merge results.
    This avoids length-limit failures from asking for one very large JSON object.
    """
    report_body, mention_rows = split_report_and_mention_table(case_report_text)

    if not mention_rows:
        raw = generate_anchor_queries(
            case_report_text,
            base_url=base_url,
            max_retries=max_retries,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            model_name=model_name,
        )
        return normalize_anchor_queries(raw, max_queries_per_uid)

    if len(mention_rows) <= chunk_size and len(case_report_text) <= max_report_chars:
        raw = generate_anchor_queries(
            case_report_text,
            base_url=base_url,
            max_retries=max_retries,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            model_name=model_name,
        )
        merged = normalize_anchor_queries(raw, max_queries_per_uid)
    else:
        merged = {}

    def process_rows(rows: List[Tuple[str, str]], depth: int = 0) -> None:
        chunk_report = build_chunk_case_report(
            report_body,
            rows,
            context_window_chars=context_window_chars,
            max_report_chars=max_report_chars,
        )
        print(f"Generating anchor queries for {len(rows)} UID mentions")
        raw = generate_anchor_queries(
            chunk_report,
            base_url=base_url,
            max_retries=max_retries,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            model_name=model_name,
        )
        normalized = normalize_anchor_queries(raw, max_queries_per_uid)

        if not normalized and len(rows) > 1:
            midpoint = len(rows) // 2
            print(f"Anchor query batch failed; splitting {len(rows)} rows into {midpoint} and {len(rows) - midpoint}")
            process_rows(rows[:midpoint], depth + 1)
            process_rows(rows[midpoint:], depth + 1)
            return

        merge_anchor_queries(merged, normalized, max_queries_per_uid)

    if not merged:
        for row_batch in batched(mention_rows, chunk_size):
            process_rows(row_batch)

    missing_rows = [
        (uid, mention)
        for uid, mention in mention_rows
        if not any(key_contains_uid(key, uid) for key in merged)
    ]

    if missing_rows:
        print(f"Adding fallback anchor queries for {len(missing_rows)} UID mentions omitted by the LLM")
        for uid, mention in missing_rows:
            merge_anchor_queries(merged, default_anchor_queries(uid, mention, max_queries_per_uid), max_queries_per_uid)

    return merged


# ------------------------------------------------------------------
# 1.  Anchor Query Generation
# ------------------------------------------------------------------
def generate_anchor_queries(case_report_text: str,
                            base_url: str = "http://localhost:8080",
                            max_retries: int = 3,
                            system_prompt_path: Path = Path("templates/aqg_system.txt"),
                            user_prompt_path: Path = Path("templates/aqg_user.txt"),
                            model_name: str = 'gpt-4o') -> Dict[str, List[str]]:
    """
    Generate anchor queries for temporal localization of discharge summary events.
    Returns a dictionary mapping events to lists of EHR database information that could help locate them.
    Retries up to max_retries times if JSON parsing fails.
    Loads system and user prompts from files if provided, falling back to defaults if missing.
    """

    # Load prompts from files if they exist, otherwise use fallback defaults
    system_prompt = read_prompt_file(system_prompt_path) or (
        "You are a clinical data analyst. Analyze the discharge summary and identify "
        "events that could be better temporally located using structured EHR data. "
        "For each event, suggest specific EHR database information that could provide "
        "temporal anchors for timing these events."
    )

    user_prompt = read_prompt_file(user_prompt_path) or (
        """
        We are trying to locate the timing of events (times relative to time of admission) in a discharge summary.
        We have an accessory EHR database with tables such as labs, inputs/outputs, medications, charting, and so on,
        that may have timestamped rows that help us identify the discharge summary event timings better.
        Given the following discharge summary, produce a complete list of clinical events from the discharge summary that could be better located
        temporally given information in the EHR database, and suggest the information in the structured data that
        might give clues to locating those events.  Each paragraph/section with a localizable event should include at least 1 anchor.
        Output your results in the following valid json format: {{ <event_1>: [<information description 1a>, <information description 1b>, ...], ... }}
        where event_1, event_2, etc, are the events that can be improved.

        E.g.:
        {{
        "development of new pneumonia during hospitalization":
         [
            "Radiology report timestamps for chest X-ray showing new pneumonia findings",
            "Antibiotic administration records (vancomycin, cefepime, flagyl) start times",
            "Nursing documentation of respiratory status changes and oxygen requirement increases"
         ],
         "creatinine of 4.2":
         [
            "Matching laboratory finding of Cr 4.2",
            "Chart event for serum blood collection after noted decreased urine output",
         ],
         "A LP was then performed to look for Leptomeningeal disease":
         [ "Lumbar puncture procedure times",
           "CSF fluid sent for laboratory analysis",
           "PCR or cytology results on cerebrospinal fluid"
         ],

        ...
        }}

        Case report:
        {case_report_txt}
        """
    )

    llm = ChatOpenAI(
        openai_api_base=f"{base_url}/v1",
        openai_api_key="dummy",
        model=model_name,
        temperature=0,
        max_tokens=32768,  # 32K tokens for long discharge summaries in large cohort
    )

    CLAIM_EVIDENCE_SCHEMA = {
        "name": "claim_evidence_map",
        "schema": {
            "type": "object",
            "additionalProperties": {
            "type": "array",
            "items": {"type": "string"}
            }
        },
        "strict": False  # Changed from True to allow partial JSON on truncation
    }
    structured_llm = llm.with_structured_output(CLAIM_EVIDENCE_SCHEMA)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])
    
    # chain = prompt | llm | StrOutputParser()
    chain = prompt | structured_llm
    # chain = prompt | llm

    for attempt in range(max_retries):
        try:
            result = chain.invoke({"case_report_txt": case_report_text})
        except OutputParserException as e:
            if attempt < max_retries - 1:
                print(f"OutputParserException on attempt {attempt + 1}/{max_retries}, retrying...")
                print(f"Error: {e}")
                continue
            else:
                print(f"Failed to get valid output after {max_retries} attempts")
                print(f"Error: {e}")
                return {}
        except LengthFinishReasonError as e:
            if attempt < max_retries - 1:
                print(f"Token limit reached on attempt {attempt + 1}/{max_retries}, retrying...")
                print(f"Error: {e}")
                continue
            else:
                print(f"Failed: Token limit exceeded after {max_retries} attempts")
                print(f"Error: {e}")
                return {}
        # import pdb
        # pdb.set_trace()
        if type(result) is dict:
            return result
        elif type(result) is AIMessage:
            print('Pulling content from result')
            result = result.content
        try:
            return parse_or_repair_json(result, llm, attempt)
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"JSON parsing failed on attempt {attempt + 1}/{max_retries}, retrying...\n(---Error---\n{e}\n--- ---\n)")
                continue
            else:
                print(f"Failed to parse JSON response from LLM after {max_retries} attempts: {e}")
                print(f"Raw response: {result}")
                return {}
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                continue
            else:
                print(f"Failed after {max_retries} attempts due to unexpected error: {e}")
                return {}
    # Fallback: return empty dict if all retries failed
    return {}


def save_anchor_queries(anchor_queries: Dict[str, List[str]], output_dir: Path, filename: str) -> None:
    """
    Save generated anchor queries to a JSON file.
    """
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename
    
    # Check if file already exists and has content
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                existing_content = json.load(f)
            # Only overwrite if file is empty or contains only empty dict
            if existing_content and existing_content != {}:
                # return (don't save), unless also:
                if ' ' in existing_content and len(existing_content)==1:
                    pass
                else:
                    print(f"Skipping {filename} - file already contains data")
                    return
        except (json.JSONDecodeError, IOError):
            # File exists but is invalid JSON or can't be read, proceed with overwrite
            pass
    
    # Clean up (possibly removing placeholder)
    try:
        with open(output_path, 'w') as f:
            json.dump(anchor_queries, f, indent=2)
    except:
        print(f'Failed to save to {output_path}')
        if output_path.exists():
            try:
                with open(output_path, 'r') as f:
                    existing_content = json.load(f)
                    if existing_content and existing_content != {}:
                        if ' ' in existing_content and len(existing_content)==1:
                            print(f'Removing placeholder at {output_path}')
                            os.unlink(f)
            except:
                pass
    
    print(f"Anchor queries saved to {output_path}")


def read_text_file(file_path: Path) -> str:
    """
    Read text from either .txt or .txt.gz files.
    """
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return f.read()
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


def process_folder(folder: Path, 
                  base_url: str = "http://localhost:8080",
                  infix: str = "anchors",
                  system_prompt_path: Path = Path("templates/aqg_system.txt"),
                  user_prompt_path: Path = Path("templates/aqg_user.txt"),
                  model_name: str = "gpt-4o",
                  chunk_size: int = 25,
                  context_window_chars: int = 700,
                  max_report_chars: int = 25000,
                  max_queries_per_uid: int = 3) -> None:
    """
    Process all text files (both .txt and .txt.gz) in folder and generate anchor queries for each.
    """
    anchors_dir = folder / infix
    anchors_dir.mkdir(exist_ok=True)
    
    # Process all .txt and .txt.gz files in the folder
    text_files = list(folder.glob("*.txt")) + list(folder.glob("*.txt.gz"))
    random.shuffle(text_files)
    for text_file in text_files:
        print(f"Processing {text_file.name}...")
        
        # Check if anchor queries file already exists and has content
        output_filename = text_file.stem + ".json"
        output_path = anchors_dir / output_filename
        
        if output_path.exists():
            try:
                with open(output_path, 'r') as f:
                    existing_content = json.load(f)
                # If file exists and has non-empty content, skip generation
                if existing_content and not is_placeholder_anchor_json(existing_content):
                    print(f"Skipping {text_file.name} - anchor queries already exist")
                    continue
                if is_placeholder_anchor_json(existing_content):
                    print(f"Existing file {output_filename} is only a placeholder, regenerating...")
            except (json.JSONDecodeError, IOError):
                # File exists but is invalid JSON or can't be read, proceed with generation
                print(f"Existing file {output_filename} is invalid, regenerating...")
        
        # Save a placeholder
        with open(output_path, 'w') as f:
            json.dump({" ": None}, f, indent=2)

        # Read the case report text
        case_report_text = read_text_file(text_file)
        
        # Generate anchor queries
        anchor_queries = generate_anchor_queries_batched(
            case_report_text,
            base_url=base_url,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            model_name=model_name,
            chunk_size=chunk_size,
            context_window_chars=context_window_chars,
            max_report_chars=max_report_chars,
            max_queries_per_uid=max_queries_per_uid,
        )
        
        # Save to file
        save_anchor_queries(anchor_queries, anchors_dir, output_filename)
        
        print(f"Generated {len(anchor_queries)} anchor queries for {text_file.name}")

# ------------------------------------------------------------------
# 2.  CLI interface
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate"])
    parser.add_argument("--folder", default=".", help="Folder containing text files to process")
    parser.add_argument("--host", default="localhost", help="Host for LLM server")
    parser.add_argument("--port", type=int, default=8080, help="Port for LLM server")
    parser.add_argument("--infix", default="anchors", help="Sub-folder name for anchor queries")
    parser.add_argument("--system-prompt", default="templates/aqg_system.txt", help="Path to system prompt file")
    parser.add_argument("--user-prompt", default="templates/aqg_user.txt", help="Path to user prompt file")
    parser.add_argument("--model-name", default="gpt-4o", help="Model name (exact as in vllm if used)")
    parser.add_argument("--chunk-size", type=int, default=25, help="UID mention rows per anchor-query LLM call")
    parser.add_argument("--context-window-chars", type=int, default=700, help="Report context characters around each UID tag")
    parser.add_argument("--max-report-chars", type=int, default=25000, help="Maximum report excerpt characters per anchor-query call")
    parser.add_argument("--max-queries-per-uid", type=int, default=3, help="Maximum generated anchor queries retained per UID")
    
    args = parser.parse_args()
    
    folder = Path(args.folder)
    base_url = f"http://{args.host}:{args.port}"
    
    if args.action == "generate":
        process_folder(
            folder,
            base_url=base_url,
            infix=args.infix,
            system_prompt_path=Path(args.system_prompt),
            user_prompt_path=Path(args.user_prompt),
            model_name=args.model_name,
            chunk_size=args.chunk_size,
            context_window_chars=args.context_window_chars,
            max_report_chars=args.max_report_chars,
            max_queries_per_uid=args.max_queries_per_uid,
        )
        print("Anchor query generation completed for all files in folder.")
