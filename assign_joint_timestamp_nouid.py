"""
Assign jointly plausible timelines to mentions using LLM reasoning at the case report level.
Outputs a JSON structure with reasoning and multiple plausible timelines.
Usage:
python assign_joint_timestamps.py \
--anchor-results ./data/joined_anchor_source.csv \
--case-report ./data/case_report.txt \
--mention-table ./data/10056200-DS_temporal.txt.gz \
--output ./data/10056200-DS_timelines.json \
--tau 0.1 \
--host localhost \
--port 8080
"""
import pandas as pd
from pathlib import Path
import argparse
import os
from typing import Dict, List, Any, Tuple, Optional, Set
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages.ai import AIMessage
import json
import re
from datetime import datetime
import gzip
import requests
from call_lcpp import query_llama_cpp  # Import the function from call_lcpp.py

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
MENTION_TABLE_RE = re.compile(r"\n\s*Mention table:\s*\n", re.IGNORECASE)
MODEL_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"
HARMONY_MESSAGE_MARKER = "<|message|>"
UID4_TAG_RE = re.compile(r"</?([0-9a-f]{4})>", re.IGNORECASE)
UID4_TOKEN_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{4})(?![0-9a-f])", re.IGNORECASE)
UID4_FIRST_COLUMN_RE = re.compile(r"(?im)^\s*<?([0-9a-f]{4})>?\s*\|")
UID_LABEL_RE = re.compile(r"\b(?:uid4|context[ _]+uid4s?)\b", re.IGNORECASE)

def read_text_file(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return f.read()

def normalized_column_name(name: Any) -> str:
    return re.sub(r"\s+", "_", str(name).strip().lower())

def uid4s_from_uid_columns(df: pd.DataFrame) -> Set[str]:
    """Collect UID4 values only from columns explicitly identified as UID columns."""
    values: Set[str] = set()
    for column in df.columns:
        if normalized_column_name(column) not in {"uid4", "context_uid4s"}:
            continue
        for value in df[column].dropna().astype(str):
            values.update(match.lower() for match in UID4_TOKEN_RE.findall(value))
    return values

def collect_known_uid4s(anchor_file: Path, case_report_file: Path,
                        mention_file: Path) -> Set[str]:
    """Collect source UID4s before sanitization so outbound data can be audited."""
    known: Set[str] = set()

    case_text = read_text_file(case_report_file)
    known.update(match.lower() for match in UID4_TAG_RE.findall(case_text))
    known.update(match.lower() for match in UID4_FIRST_COLUMN_RE.findall(case_text))

    mention_df = pd.read_csv(
        mention_file, sep="|", compression="infer", dtype=str
    )
    known.update(uid4s_from_uid_columns(mention_df))

    anchor_df = pd.read_csv(
        anchor_file, sep=",", compression="infer", dtype=str
    )
    known.update(uid4s_from_uid_columns(anchor_df))
    for column in ("anchor", "event", "line_level_event", "line_level_value"):
        if column not in anchor_df.columns:
            continue
        for value in anchor_df[column].dropna().astype(str):
            known.update(match.lower() for match in UID4_TAG_RE.findall(value))

    return known

def remove_known_uid4s(text: str, known_uid4s: Set[str]) -> str:
    """Remove known patient UID4 values, including untagged occurrences."""
    known = {uid.lower() for uid in known_uid4s}

    return UID4_TOKEN_RE.sub(
        lambda match: (
            "[REMOVED]"
            if match.group(1).lower() in known
            else match.group(0)
        ),
        str(text),
    )

def audit_model_input(rendered_messages: List[Dict[str, Any]],
                      known_uid4s: Set[str],
                      log_file: Optional[Path] = None) -> None:
    """Fail closed if a known UID or UID field/tag remains in the model input."""
    serialized = json.dumps(rendered_messages, ensure_ascii=False)
    serialized_lower = serialized.lower()
    leaked = {
        uid for uid in known_uid4s
        if re.search(
            rf"(?<![0-9a-f]){re.escape(uid)}(?![0-9a-f])",
            serialized_lower,
            re.IGNORECASE,
        )
    }
    has_uid_tag = UID4_TAG_RE.search(serialized) is not None
    has_uid_label = UID_LABEL_RE.search(serialized) is not None

    if leaked or has_uid_tag or has_uid_label:
        # Do not print the UID values or save the unsafe prompt.
        raise RuntimeError(
            "UID audit failed before model invocation: "
            f"known_uid_matches={len(leaked)}, "
            f"uid_tag_present={has_uid_tag}, uid_label_present={has_uid_label}"
        )

    print(
        "UID audit passed before model invocation: "
        f"checked {len(known_uid4s)} known UID4 values"
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "uid_audit": "passed",
                    "known_uid4_count": len(known_uid4s),
                    "messages": rendered_messages,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved UID-audited model input to {log_file}")

def read_case_report(case_report_file: Path) -> str:
    """Read the case report text from file, handling both .txt and .txt.gz files."""
    if case_report_file.suffix == '.gz':
        with gzip.open(case_report_file, 'rt', encoding='utf-8') as f:
            text = f.read()
            text = remove_uid_tags_in_case_report(text)
            return remove_uid4_column_in_case_report(text)
    else:
        with open(case_report_file, 'r', encoding='utf-8') as f:
            text = f.read()
            text = remove_uid_tags_in_case_report(text)
            return remove_uid4_column_in_case_report(text)

def remove_uid_tags_in_case_report(text: str) -> str:
    return re.sub(r"<[^<>]*>", "", text).strip()

def remove_uid4_column_in_case_report(text: str) -> str:
    text = re.sub(r"(?im)^\s*uid4\s*\|\s*", "", text)
    return re.sub(r"(?im)^\s*<?[0-9a-f]{4}>?\s*\|\s*", "", text)

def read_mention_table(mention_file: Path) -> str:
    """Read mention table from .bsv.gz file."""
    df = pd.read_csv(
        mention_file,
        sep="|",
        compression="infer"
    )
    # Forget the UID4 tags
    uid_columns = [
        column for column in df.columns
        if normalized_column_name(column) in {"uid4", "context_uid4s"}
    ]
    df = df.drop(
        columns=uid_columns,
        errors="ignore"
    )
    return df.to_csv(sep="|", index=False)

def read_anchor_table(anchor_file: Path) -> pd.DataFrame:
    """Read an anchor table from an uncompressed or gzip-compressed CSV file."""
    df = pd.read_csv(
        anchor_file,
        sep=",",
        compression="infer"
    )
    # Remove any mention of a UID4 tag
    for column in ("anchor", "event", "line_level_event", "line_level_value"):
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.replace(r"<[^<>]*>", "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
    return df

def extract_note_id_from_case_report_path(case_report_file: Path) -> str:
    """Extract note_id from the case report file path."""
    filename = case_report_file.name
    if filename.endswith('.txt.gz'):
        return filename[:-7]
    elif filename.endswith('.txt'):
        return filename[:-4]
    else:
        return filename.split('.')[0]

def candidate_ids_from_name(name: str) -> List[str]:
    """Return possible note/subject IDs from a bare or decorated filename."""
    base_name = Path(str(name)).name
    for suffix in (
        '.csv.gz', '.json.gz', '.txt.gz', '.bsv.gz',
        '.csv', '.json', '.txt', '.bsv',
    ):
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]

    for suffix in ('_summary', '_results', '_joined', '_timelines', '_extracted', '_tags'):
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]

    candidates = []

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    add(base_name)
    if base_name.endswith('.txt'):
        add(base_name[:-4])
    for token in re.findall(r"(?<!\d)\d{5,}(?!\d)", base_name):
        add(token)

    return candidates

def convert_timestamp_to_datetime(timestamp_int: int) -> str:
    """Convert integer timestamp (Unix time) to formatted datetime string in UTC."""
    try:
        dt = datetime.utcfromtimestamp(timestamp_int)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(timestamp_int)

def read_note_link_times(note_link_file: Path, note_id: str) -> Dict[str, str]:
    """Read note_link.csv.gz and extract admission/discharge times."""
    if not note_link_file.exists():
        return {}
    try:
        df = pd.read_csv(
            note_link_file,
            compression='gzip' if note_link_file.suffix == '.gz' else None,
            dtype={'note_id': str, 'hadm_id': str, 'subject_id': str},
        )
        if 'note_id' not in df.columns:
            print(f"Error reading note_link: missing note_id column in {note_link_file}")
            return {}

        for column in ('note_id', 'hadm_id', 'subject_id'):
            if column in df.columns:
                df[column] = df[column].astype(str)

        for lookup_note_id in candidate_ids_from_name(note_id):
            matching_row = df[df['note_id'] == lookup_note_id]
            if matching_row.empty and 'subject_id' in df.columns:
                matching_row = df[df['subject_id'] == lookup_note_id]
            if matching_row.empty:
                matching_row = df[df['note_id'].str.startswith(f"{lookup_note_id}-")]
            if not matching_row.empty:
                return {
                    'admittime': matching_row.iloc[0]['admittime'],
                    'dischtime': matching_row.iloc[0]['dischtime']
                }

        return {}
    except Exception as e:
        print(f"Error reading note_link: {e}")
        return {}

def truncate_text(value: Any, max_chars: int) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text

def strip_embedded_mention_table(case_report: str) -> str:
    """Bundle reports already append a mention table; Step 6 adds it separately."""
    return MENTION_TABLE_RE.split(case_report, maxsplit=1)[0].strip()

def compact_case_report(case_report: str,
                        mention_table: str,
                        max_chars: int,
                        context_chars: int = 240) -> str:
    """Keep local context around uid4 tags, falling back to a front slice."""
    report_body = strip_embedded_mention_table(case_report)
    if max_chars <= 0 or len(report_body) <= max_chars:
        return report_body

    snippets = []

    if snippets:
        return "\n".join(snippets)[:max_chars]

    return report_body[:max_chars]

def prepare_candidates_data(anchor_df: pd.DataFrame,
                            tau: float,
                            max_total_candidates: float = 300,
                            top_k: int = 10,
                            max_field_chars: int = 180) -> List[Dict]:
    """Prepare candidate data for LLM from anchor results."""
    filtered_df = anchor_df[anchor_df['score'] >= tau].copy()
    filtered_df = (
        filtered_df.sort_values(['anchor', 'score'], ascending=[True, False])
        .groupby('anchor')
        .head(top_k)
        .reset_index(drop=True)
    )
    filtered_df = filtered_df.head(max_total_candidates)
    
    candidates = []
    for rowi, row in filtered_df.iterrows():
        # Extract anchor text from query
        anchor_text = re.search(r"Information regarding:\s*(.*?)(?:, in particular:|$)", row['anchor'])
        if anchor_text:
            anchor_text = anchor_text.group(1).strip()
        else:
            anchor_text = row['anchor']
        
        candidates.append({
            'anchor': truncate_text(re.sub(r'<[^>]+>', '', anchor_text), max_field_chars),
            'score': f"{round(float(row['score']),3):.3f}",
            'timestamp': convert_timestamp_to_datetime(row['line_level_timestamp']),
            'event': truncate_text(row['line_level_event'], max_field_chars),
            'value': truncate_text(row['line_level_value'], max_field_chars)
        })
    return candidates

def query_openai_chat(messages: List[Dict], server_url: str, max_tokens: int = 100000) -> str:
    """Query OpenAI-compatible chat completions endpoint."""
    url = f"{server_url}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": min(max_tokens, 120000),  # Cap to avoid server issues
        "temperature": 0.7,
        "repeat_penalty": 1.18,  # These are also in query_llama_cpp
        "repeat_last_n": 256,
        "presence_penalty": 0.6
    }
    try:
        r = requests.post(url, json=payload)
        if r.status_code == 200:
            response_data = json.loads(r.text)
            return response_data['choices'][0]['message']['content']
        else:
            print(f"OpenAI endpoint error: {r.status_code}")
            return ""
    except Exception as e:
        print(f"Error querying OpenAI endpoint: {e}")
        return ""

def strip_code_fences(text: str) -> str:
    text = text.strip()
    m = FENCE_RE.search(text)
    return m.group(1).strip() if m else text

def extract_balanced_json_object(text: str) -> str:
    """Extract the first balanced JSON object from a model response."""
    start = text.find("{")
    if start < 0:
        return text.strip()

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1].strip()

    last = text.rfind("}")
    if last > start:
        return text[start:last + 1].strip()
    return text[start:].strip()

def clean_model_json_text(result: Any) -> str:
    """Strip wrappers commonly emitted by local OpenAI-compatible models."""
    if isinstance(result, AIMessage):
        result = result.content

    text = str(result).strip()

    # GPT-OSS/Harmony-style responses can wrap the final answer in channel tokens.
    if HARMONY_FINAL_MARKER in text:
        text = text.rsplit(HARMONY_FINAL_MARKER, 1)[-1]
    elif HARMONY_MESSAGE_MARKER in text:
        text = text.rsplit(HARMONY_MESSAGE_MARKER, 1)[-1]

    text = MODEL_SPECIAL_TOKEN_RE.sub("", text).strip()
    text = strip_code_fences(text)
    if text.lower().startswith("json\n"):
        text = text.split("\n", 1)[1].strip()

    return extract_balanced_json_object(text)

def parse_or_repair_json(result: str, llm, attempt: int):
    text = clean_model_json_text(result)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(text + "}")
        except:
            pass
        pass

    print(f"JSON parsing llm try {attempt+1}")
    # CLAIM_EVIDENCE_SCHEMA = {
    #     "name": "timeline_reasoning",
    #     "schema": {
    #         "type": "object",
    #         "additionalProperties": {
    #             "type": "array",
    #             "items": {"type": "string"}
    #         }
    #     },
    #     "strict": True
    # }
    # structured_llm = llm.with_structured_output(CLAIM_EVIDENCE_SCHEMA)
    
    # chain = prompt | llm | StrOutputParser()

    repair_grounding = ChatPromptTemplate.from_messages([
        ("system", "You are a data engineer."),
        ("user", "Fix this so it is valid JSON.\nReturn ONLY JSON.\n\n{bad_json}"),
    ])

    repair_chain = repair_grounding | llm | StrOutputParser()
    repair_result = clean_model_json_text(repair_chain.invoke({"bad_json": text}))

    return json.loads(repair_result)

def has_meaningful_content(llm_structured_output,
        *,
        min_rows: int = 2,
        min_cols: int = 2,
        min_chars: int = 20,
) -> Tuple[bool, Any]:
    """
    Validate LLM output.

    Returns:
        should_retry (bool)
        parsed_output (Any | None)
    """
    if not isinstance(llm_structured_output, dict) or not llm_structured_output:
        return False, llm_structured_output

    if 'timelines' not in llm_structured_output.keys() or 'reasoning_summary' not in llm_structured_output.keys():
        return False, llm_structured_output

    parsed = llm_structured_output['timelines']

    if not isinstance(parsed, list) or not parsed:
        return False, llm_structured_output

    for item in parsed:
        if not isinstance(item, dict):
            return False, llm_structured_output

        data = item.get("data", "")

        # Reject trivial placeholders like "," or ""
        if not isinstance(data, str) or len(data.strip()) < min_chars:
            return False, llm_structured_output

        lines = [ln for ln in data.splitlines() if ln.strip()]
        if len(lines) < min_rows:
            return False, llm_structured_output

        # BSV heuristic: first non-empty line must contain | separators
        header = lines[0]

        # Reject if using tab separators (should be pipe-separated BSV)
        if '\t' in header:
            print(f"ERROR: Timeline data uses tab separators instead of pipe separators (BSV)")
            print(f"Header: {repr(header[:100])}")
            return False, llm_structured_output

        cols = [c for c in header.split("|") if c.strip()]
        if len(cols) < min_cols:
            print(f"ERROR: Timeline data has too few columns (expected pipe-separated BSV)")
            print(f"Found {len(cols)} columns: {cols}")
            return False, llm_structured_output

    # Passed all checks
    return True, llm_structured_output

def query_llm_for_joint_timelines(case_report: str, mention_table: str, 
                                candidates: List[Dict], admit_time: str, 
                                discharge_time: str, base_url: str, template_path: Path, 
                                endpoint_type: str = "openai",
                                num_timelines: int = 1,
                                max_case_report_chars: int = 30000,
                                case_context_chars: int = 240,
                                known_uid4s: Optional[Set[str]] = None,
                                model_input_log_file: Optional[Path] = None) -> Dict:
    """Use LLM to determine jointly plausible timelines."""
    
    # Read template file
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            user_text = f.read()
    except FileNotFoundError:
        print(f"Template file {template_path} not found, using default template")
        # Fallback to original template if file doesn't exist
        user_text = """
Case Report:
{{case_report}}
Current Mention Table:
{{mentions_text}}
Candidate Timestamps (score >= threshold):
{{candidates_text}}
Hospitalization Context:
- Admission: {{admit_time}}
- Discharge: {{discharge_time}}
Provide 2-3 jointly plausible timelines where all timestamps are consistent.
For each timeline, provide:
- Updated mention table with new timestamps (in similar format with header)
- Reasoning for the timeline's plausibility
- Any tensions in timing given the evidence
Output format:
{{
    "reasoning_summary": "overall clinical reasoning...",
    "timelines": [
        {{
            "timeline_id": 1,
            "data": "revised bsv table as a string",
            "reasoning": "what makes this timeline plausible and complementary..."
        }},
        ...
    ]
}}
"""
    
    # Format mentions text
    mentions_text = mention_table
    case_report = compact_case_report(
        case_report,
        mentions_text,
        max_case_report_chars,
        context_chars=case_context_chars,
    )

    # Format candidates
    candidates_text = json.dumps(candidates, indent=2)

    # Escape curly braces in all input text to prevent ChatPromptTemplate interpretation
    # This must be done BEFORE replacing placeholders
    case_report_escaped = case_report.replace('{', '{{').replace('}', '}}')
    mentions_text_escaped = mentions_text.replace('{', '{{').replace('}', '}}')
    candidates_text_escaped = candidates_text.replace('{', '{{').replace('}', '}}')
    admit_time_escaped = admit_time.replace('{', '{{').replace('}', '}}')
    discharge_time_escaped = discharge_time.replace('{', '{{').replace('}', '}}')

    # Replace placeholders in template with escaped content
    user_text = user_text.replace("{{case_report}}", case_report_escaped)
    user_text = user_text.replace("{{mentions_text}}", mentions_text_escaped)
    user_text = user_text.replace("{{candidates_text}}", candidates_text_escaped)
    user_text = user_text.replace("{{admit_time}}", admit_time_escaped)
    user_text = user_text.replace("{{discharge_time}}", discharge_time_escaped)
    user_text = user_text.replace("{{num_timelines}}", str(max(1, int(num_timelines))))
    
    # if endpoint_type == "openai":
    if True:
        # For OpenAI chat completions, return structured message
        messages = [
            {"role": "system", "content": "You are a top-shelf clinician. Be concise. Return only the requested JSON. Do not skip any rows from the Current Mention Table."},
            {"role": "user", "content": user_text}
        ]
        
        # response = ''
        # for wsi in range(10):
        #     try:
        #         response = query_openai_chat(messages, server_url=base_url)
        #         if response.strip():
        #             break
        #     except:
        #         print('Process error (ctx window?)')
        #         response = ''
        #         break
        #     print(f'Whitespace output ({wsi}/10)')

        llm = ChatOpenAI(
            openai_api_base=f"{base_url}/v1",
            openai_api_key="dummy",
            model="/data/weissjc/.cache/vllm/Kimi-K2-Thinking/",
            # temperature=0,
            max_tokens=100000,
        )   
        messages.append({
            "role": "user",
            "content": (
                "Return ONLY one valid JSON object. Do not include markdown fences, "
                "prose, analysis outside the JSON, or model-specific channel markers. "
                "The JSON must contain keys 'reasoning_summary' and 'timelines'. "
                "Each timeline must contain 'timeline_id', 'data', and 'reasoning'."
            )
        })

        # Remove tagged or untagged known UID4 values from everything
        # that is about to be sent to the model.
        messages = [
            {
                **message,
                "content": remove_known_uid4s(
                    message["content"],
                    known_uid4s or set(),
                ),
            }
            for message in messages
        ]

        try:
            prompt = ChatPromptTemplate.from_messages(messages)
        except (KeyError, ValueError) as e:
            print(f"ERROR: ChatPromptTemplate parsing error: {e}")
            print(f"       This usually means input data contains curly braces that look like template variables")
            print(f"       Returning empty result for this case")
            return {"timelines": [], "error": str(e)}

        prompt_value = prompt.invoke({})
        rendered_messages = [
            {"type": message.type, "content": message.content}
            for message in prompt_value.to_messages()
        ]
        audit_model_input(
            rendered_messages,
            known_uid4s or set(),
            model_input_log_file,
        )

        chain = prompt | llm | StrOutputParser()

        result_text = ""
        try:
            result_text = chain.invoke({})
            return parse_or_repair_json(result_text, llm, 0)
        except Exception as e:
            print(f"ERROR: LLM response could not be parsed as timeline JSON: {e}")
            return {
                "timelines": [],
                "error": str(e),
                "raw_response_excerpt": str(result_text)[:4000],
            }
        # return parse_or_repair_json(response, llm, 0)        
        # # Extract JSON from response
        # if '```json' in response:
        #     json_start = response.find('```json') + len('```json')
        #     json_end = response.find('```', json_start)
        #     if json_end != -1:
        #         response = response[json_start:json_end].strip()
        # elif '```' in response:
        #     json_start = response.find('```') + len('```')
        #     json_end = response.find('```', json_start)
        #     if json_end != -1:
        #         response = response[json_start:json_end].strip()
        
        # return json.loads(response)
    else:  # completion endpoint
        prompt = f"""You are a top-shelf, underrated clinician at the pinnacle of your ability. Be concise and comprehensive. Do not skip any rows.

Case Report (with uid4 tags):
{case_report}

Current Mention Table:
{mentions_text}

Candidate Timestamps (score >= threshold):
{json.dumps(candidates, indent=2)}

Hospitalization Context:
- Admission: {admit_time}
- Discharge: {discharge_time}

Provide 2-3 jointly plausible timelines where all timestamps are consistent.
For each timeline, provide:
- Updated mention table with new timestamps (in similar format with header)
- Reasoning for the timeline's plausibility
- Any tensions in timing given the evidence
Output format:
{{
    "reasoning_summary": "overall clinical reasoning...",
    "timelines": [
        {{
            "timeline_id": 1,
            "data": "put revised bsv table here",
            "reasoning": "what makes this timeline plausible and complementary..."
        }},
        ...
    ]
}}"""
        
        response = ''
        for wsi in range(10):
            try:
                response = query_llama_cpp(prompt, server_url=f"{base_url}/completion", n_predict=200000,
                                         extra_args = {
                                             "repeat_penalty": 1.18, 
                                             "repeat_last_n": 256,
                                             "presence_penalty": 0.6
                                         })
                if response.strip():
                    break
            except:
                print('Process error (ctx window?)')
                response = ''
                break
            print(f'Whitespace output ({wsi}/10)')
        
        # # Extract JSON from response
        # if '```json' in response:
        #     json_start = response.find('```json') + len('```json')
        #     json_end = response.find('```', json_start)
        #     if json_end != -1:
        #         response = response[json_start:json_end].strip()
        # elif '```' in response:
        #     json_start = response.find('```') + len('```')
        #     json_end = response.find('```', json_start)
        #     if json_end != -1:
        #         response = response[json_start:json_end].strip()
        
        # return json.loads(response)

        llm = ChatOpenAI(
            openai_api_base=f"{base_url}/v1",
            openai_api_key="dummy",
            model="/data/weissjc/.cache/vllm/Kimi-K2-Thinking/",
            temperature=0,
            max_tokens=100000,
        )   

        STRUCTURED_SCHEMA = {
            "title": "TimeSeriesResponse",
            "description": "LLM output containing an overall reasoning summary and one or more named time series with tabular data and per-series reasoning.",
            "type": "object",
            "required": ["reasoning_summary", "time_series"],
            "properties": {
                "reasoning_summary": {"type": "string"},
                "time_series": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["timeline_id", "data", "reasoning"],
                    "properties": {
                        "timeline_id": {"type": "string", "minLength": 1},
                        "data": {"type": "string"},
                        "reasoning": {"type": "string"}
                    },
                    "additionalProperties": False
                }
                }
            },
            "additionalProperties": False
        }
        prompt = ChatPromptTemplate.from_messages(messages)
        
        structured_llm = llm.with_structured_output(STRUCTURED_SCHEMA)
        # return parse_or_repair_json(response, structured_llm, 0)  
        chain = prompt | structured_llm 
        result = chain.invoke({})
        if result is Dict:
            return result
        if type(result) is AIMessage:
            print('Pulling content from result')
            result = result.content
        return parse_or_repair_json(result, structured_llm, 0)   

def log_failed_output(
    case_file: str,
    log_dir: str,
    attempt: int,
    raw_output: str,
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    basename= os.path.basename(case_file).replace(".*","")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(log_dir, f"{basename}_retry_{attempt}_{ts}.json")

    with open(path, "w", encoding="utf-8") as f:
        f.write(raw_output)

def assign_joint_timelines(anchor_results_file: Path, case_report_file: Path, 
                         mention_file: Path, output_file: Path, 
                         log_dir: Path,
                         tau: float,
                         base_url: str, note_link_file: Path,
                         max_total_candidates: float = 300,
                         template_path: Path = Path("templates/ajt.txt"),
                         top_k: int = 10,
                         n_retries: int = 5,
                         endpoint_type: str = "openai",
                         num_timelines: int = 1,
                         max_candidate_field_chars: int = 180,
                         max_case_report_chars: int = 30000,
                         case_context_chars: int = 240,
                         model_input_log_dir: Optional[Path] = None
                         ) -> Dict:
    """Process all inputs to generate jointly plausible timelines."""
    # Capture source UID4 values before any sanitization. These values are used
    # only for a fail-closed audit immediately before the model invocation.
    known_uid4s = collect_known_uid4s(
        anchor_results_file, case_report_file, mention_file
    )

    # Load anchor results
    anchor_df = read_anchor_table(anchor_results_file)
    
    # Read case report
    case_report = read_case_report(case_report_file)
    
    # Read mention table
    mention_table = read_mention_table(mention_file)
    
    # Extract note_id
    note_id = extract_note_id_from_case_report_path(case_report_file)

    model_input_log_file = None
    if model_input_log_dir is not None:
        model_input_log_file = (
            model_input_log_dir / f"{note_id}_model_input.json"
        )
    
    # Get hospitalization context
    note_times = read_note_link_times(note_link_file, note_id)
    
    # Prepare candidates with top-k filtering per mention
    candidates = prepare_candidates_data(
        anchor_df,
        tau,
        max_total_candidates=max_total_candidates,
        top_k=top_k,
        max_field_chars=max_candidate_field_chars,
    )
    
    # Query LLM for timelines
    result = {}
    result_passes = False
    for i in range(n_retries):        
        result = query_llm_for_joint_timelines(
            case_report, mention_table, candidates, 
            note_times.get('admittime', 'Unknown'),
            note_times.get('dischtime', 'Unknown'),
            base_url,
            template_path,
            endpoint_type,
            num_timelines=num_timelines,
            max_case_report_chars=max_case_report_chars,
            case_context_chars=case_context_chars,
            known_uid4s=known_uid4s,
            model_input_log_file=model_input_log_file,
        )

        result_passes, _ = has_meaningful_content(result)
        if result_passes:
            break
        
        # Else retry
        log_failed_output(case_report_file, log_dir, i+1, json.dumps(result))

        print(f'LLM content did not pass checks, logged to {log_dir} ({str(i+1)}/{n_retries})')

    if not result_passes:
        raise RuntimeError(
            f"LLM did not return a valid pipe-separated timeline table after {n_retries} attempts"
        )

    # if type(result) is not dict:
    #     result = dict(result)
    
    # Add metadata
    result['note_id'] = note_id
    result['tau'] = tau
    result['candidate_count'] = len(candidates)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    return result

def main():
    parser = argparse.ArgumentParser(description="Generate jointly plausible timelines")
    parser.add_argument("--anchor-results", required=True, help="Joined anchor-source CSV file")
    parser.add_argument("--case-report", required=True, help="Case report text file")
    parser.add_argument("--mention-table", required=True, help="Mention table .txt.gz file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--log-dir", help="Log file path", default="log")
    parser.add_argument("--tau", type=float, default=0.1, help="Score threshold")
    parser.add_argument("--host", default="localhost", help="LLM server host")
    parser.add_argument("--port", type=int, default=8080, help="LLM server port")
    parser.add_argument("--max-total-candidates", type=int, default=300, help="How many candidates max to consider")
    parser.add_argument("--top-k", type=int, default=10, help="Top K scores per mention to keep")
    parser.add_argument("--num-timelines", type=int, default=1, help="Number of final timelines to request")
    parser.add_argument("--max-candidate-field-chars", type=int, default=180, help="Maximum characters kept for each candidate anchor/event/value field")
    parser.add_argument("--max-case-report-chars", type=int, default=30000, help="Maximum case-report context characters sent to the LLM; <=0 keeps full report")
    parser.add_argument("--case-context-chars", type=int, default=240, help="Characters of case-report context kept around each uid4 tag")
    parser.add_argument("--note-link", default="./data/note_link.csv.gz", help="Path to note_link.csv.gz")
    parser.add_argument("--template", default="templates/ajt.txt", help="Path to template file")
    parser.add_argument(
        "--model-input-log-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for exact UID-audited model-input JSON logs. "
            "These logs still contain case-report text and must be protected."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--endpoint-type", choices=["openai", "completion"], default="openai", 
                       help="Which endpoint type to use: 'openai' for /v1/chat/completions or 'completion' for /completion (disabled for now)")
    
    args = parser.parse_args()
    
    # anchor,summary,score,bestscore,event,summary_id,row_index,line_level_timestamp,line_level_event,line_level_value,hadm_id,cutoff_time,note_id
    anchor_results = Path(args.anchor_results)
    # Output from 5A: txt files within bundle
    case_report = Path(args.case_report)
    # uid4|mention|time|bounds|known|context uid4s
    mention_table = Path(args.mention_table)
    output_file = Path(args.output)
    log_dir = Path(args.log_dir)
    template_path = Path(args.template)
    max_total_candidates = args.max_total_candidates
    
    if not all([anchor_results.exists(), case_report.exists(), mention_table.exists()]):
        print("Missing required input files")
        return 2
    
    if output_file.exists() and not args.overwrite:
        print("Output exists, use --overwrite to regenerate")
        return 0
        # pass
    
    result = assign_joint_timelines(
        anchor_results, case_report, mention_table,
        output_file,
        log_dir,
        args.tau,
        f"http://{args.host}:{args.port}",
        Path(args.note_link),
        max_total_candidates,
        template_path,
        top_k=args.top_k,
        endpoint_type=args.endpoint_type,
        num_timelines=args.num_timelines,
        max_candidate_field_chars=args.max_candidate_field_chars,
        max_case_report_chars=args.max_case_report_chars,
        case_context_chars=args.case_context_chars,
        model_input_log_dir=args.model_input_log_dir,
    )

    print(f"Generated {len(result.get('timelines', []))} timelines")
    print(f"Saved to {output_file}")
    return 0

if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main() or 0
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # Ensure we always force exit, even if background client threads are alive.
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        import os
        os._exit(exit_code)