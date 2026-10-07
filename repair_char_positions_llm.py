#!/usr/bin/env python3
"""Repair timeline character positions with a locally served language model.

The model selects exact quotes from the clinical note. This script, rather
than the model, converts those quotes to validated Python half-open offsets.
All non-position columns are preserved verbatim.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, TextIO


NULL_POSITION = "--"
TOKEN_RE = re.compile(r"[a-z0-9]+")
PATIENT_ID_RE = re.compile(r"(?<!\d)\d{5,}(?!\d)")


def open_text(path: Path, mode: str = "rt") -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode.removesuffix("t"), encoding="utf-8", newline="")


def normalized_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def spans_are_compatible(mention: str, selected_text: str) -> bool:
    """Conservatively decide whether an existing span is already plausible."""
    mention_norm = " ".join(normalized_tokens(mention))
    selected_norm = " ".join(normalized_tokens(selected_text))
    if not mention_norm or not selected_norm:
        return False
    if mention_norm in selected_norm or selected_norm in mention_norm:
        return True
    mention_tokens = set(mention_norm.split())
    selected_tokens = set(selected_norm.split())
    return len(mention_tokens & selected_tokens) / max(1, len(mention_tokens)) >= 0.6


def parse_position(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.isdigit() else None


def patient_ids(path: Path) -> set[str]:
    return set(PATIENT_ID_RE.findall(path.name))


def current_span(row: dict[str, str], note: str) -> tuple[int, int, str] | None:
    start = parse_position(row.get("char_pos", ""))
    end = parse_position(row.get("char_pos_ub", ""))
    if start is None or end is None or not (0 <= start < end <= len(note)):
        return None
    return start, end, note[start:end]


def annotated_note(note: str) -> str:
    """Prefix every note line with its exact zero-based character offset."""
    lines: list[str] = []
    cursor = 0
    for line in note.splitlines(keepends=True):
        lines.append(f"[{cursor:08d}] {line.rstrip(chr(10) + chr(13))}")
        cursor += len(line)
    if not note or (note and note[-1] not in "\r\n"):
        # splitlines(keepends=True) already emitted the final non-newline line.
        return "\n".join(lines)
    if cursor < len(note):
        lines.append(f"[{cursor:08d}] {note[cursor:]}")
    return "\n".join(lines)


def row_request(row: dict[str, str], note: str) -> dict[str, Any]:
    existing = current_span(row, note)
    return {
        "uid4": row.get("uid4", "").strip(),
        "mention": row.get("mention", "").strip(),
        "current_start": existing[0] if existing else None,
        "current_end": existing[1] if existing else None,
        "current_text": existing[2] if existing else None,
    }


def build_prompt(note: str, rows: list[dict[str, str]]) -> str:
    requests = [row_request(row, note) for row in rows]
    return f"""Repair character spans in a clinical timeline using the clinical note below.

For every requested uid4, identify the exact supporting text in the note. The timeline
mention may be normalized (for example, `male` may be written as `Sex: M`). Do not use
the current position when it points to unrelated text. Do not infer a span when the note
does not support the event.

Return ONLY a JSON array. Return exactly one object per uid4, in the same order:
{{"uid4":"...","evidence":"an exact, preferably unique surrounding quote from the note",
  "span":"the exact substring inside evidence to highlight","start":123,"end":130}}

Offsets are zero-based and end-exclusive. `evidence` and `span` must be copied exactly,
including capitalization and punctuation. Use null for evidence, span, start, and end if
there is no support. The script will independently locate and validate every quote.

ROWS:
{json.dumps(requests, ensure_ascii=False, indent=2)}

CLINICAL NOTE WITH LINE-START OFFSETS:
{annotated_note(note)}
"""


def endpoint_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def ensure_local_endpoint(base_url: str, allow_remote: bool) -> None:
    hostname = urllib.parse.urlparse(base_url).hostname
    if hostname not in {"127.0.0.1", "localhost", "::1"} and not allow_remote:
        raise ValueError(
            "Refusing to send a clinical note to a non-local endpoint. "
            "Use --allow-remote-endpoint only for an approved private endpoint."
        )


def health_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))


def server_is_healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(health_url(base_url), timeout=2) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def allocated_gpu_count() -> int:
    value = os.environ.get("SLURM_GPUS_ON_NODE", "")
    return int(value) if value.isdigit() and int(value) > 0 else 1


def start_vllm_server(
    base_url: str,
    model_path: Path,
    served_model_name: str,
    tensor_parallel_size: int,
    startup_timeout: int,
    extra_args: str,
    log_path: Path,
) -> tuple[subprocess.Popen[bytes] | None, TextIO | None]:
    """Start a local vLLM server when the requested endpoint is not healthy."""
    if server_is_healthy(base_url):
        print(f"using existing model server at {base_url}", file=sys.stderr)
        return None, None

    parsed = urllib.parse.urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("automatic server startup requires a loopback --base-url")
    if not model_path.is_dir():
        raise ValueError(f"model directory does not exist: {model_path}")
    vllm = shutil.which("vllm")
    if not vllm:
        raise ValueError("vllm executable not found in PATH; activate the vLLM environment")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    command = [
        vllm,
        "serve",
        str(model_path),
        "--served-model-name",
        served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--trust-remote-code",
        *shlex.split(extra_args),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    print(f"starting MiMo server; log={log_path}", file=sys.stderr)
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(
                f"vLLM exited with status {process.returncode}; inspect {log_path}"
            )
        if server_is_healthy(base_url):
            print("MiMo server is ready", file=sys.stderr)
            return process, log_handle
        time.sleep(5)
    stop_vllm_server(process, log_handle)
    raise TimeoutError(f"vLLM did not become healthy within {startup_timeout}s; inspect {log_path}")


def stop_vllm_server(process: subprocess.Popen[bytes], log_handle: TextIO | None) -> None:
    """Terminate the vLLM process group started by this script."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    if log_handle and not log_handle.closed:
        log_handle.close()


def query_model(
    prompt: str,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You locate exact evidence in clinical notes and return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint_url(base_url),
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content")
            if not content:
                raise ValueError("model response contained no content")
            return content
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
            if attempt >= retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def extract_json_array(response: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(response):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(response[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    raise ValueError("model response did not contain a JSON object array")


def occurrences(text: str, value: str) -> list[int]:
    if not value:
        return []
    result: list[int] = []
    cursor = 0
    while True:
        index = text.find(value, cursor)
        if index < 0:
            return result
        result.append(index)
        cursor = index + 1


def resolve_suggestion(
    suggestion: dict[str, Any], note: str
) -> tuple[int, int, str] | None:
    evidence = suggestion.get("evidence")
    span = suggestion.get("span")
    if not isinstance(span, str) or not span:
        return None
    if not isinstance(evidence, str) or not evidence:
        evidence = span

    candidates: list[tuple[int, int]] = []
    for evidence_start in occurrences(note, evidence):
        relative_starts = occurrences(evidence, span)
        candidates.extend(
            (evidence_start + relative, evidence_start + relative + len(span))
            for relative in relative_starts
        )
    candidates = list(dict.fromkeys(candidates))

    proposed_start = suggestion.get("start")
    proposed_end = suggestion.get("end")
    if isinstance(proposed_start, int) and isinstance(proposed_end, int):
        proposed = (proposed_start, proposed_end)
        if (
            0 <= proposed_start < proposed_end <= len(note)
            and note[proposed_start:proposed_end] == span
            and (not candidates or proposed in candidates)
        ):
            return proposed_start, proposed_end, span

    if len(candidates) == 1:
        start, end = candidates[0]
        return start, end, note[start:end]

    global_candidates = occurrences(note, span)
    if len(global_candidates) == 1:
        start = global_candidates[0]
        return start, start + len(span), span
    return None


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="|")
        if not reader.fieldnames:
            raise ValueError(f"timeline has no header: {path}")
        header = [column.strip() for column in reader.fieldnames]
        rows = []
        for raw in reader:
            rows.append({(key or "").strip(): (value or "").strip() for key, value in raw.items()})
    required = {"uid4", "mention"}
    if not required.issubset(header):
        raise ValueError(f"timeline must contain {sorted(required)}: {path}")
    insert_at = header.index("uid4") + 1
    for column in ("char_pos", "char_pos_ub"):
        if column not in header:
            header.insert(insert_at, column)
        insert_at = header.index(column) + 1
    return header, rows


def write_table(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=header, delimiter="|", lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def chunks(items: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair BSV character positions with an exact-quote LLM workflow."
    )
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--note", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="mimo-charpos")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/grimalditersans/models/MiMo-V2.6-Pro-RL/"),
        help="Local model directory used to start vLLM automatically",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=allocated_gpu_count())
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument(
        "--vllm-extra-args",
        default="",
        help="Additional arguments passed to `vllm serve`",
    )
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--repair-all", action="store_true", help="Ask the model to review every row")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-remote-endpoint", action="store_true")
    parser.add_argument("--allow-id-mismatch", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    for path in (args.timeline, args.note):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output exists (use --overwrite): {args.output}")
    timeline_ids = patient_ids(args.timeline)
    note_ids = patient_ids(args.note)
    if (
        timeline_ids and note_ids and timeline_ids.isdisjoint(note_ids)
        and not args.allow_id_mismatch
    ):
        parser.error(
            f"timeline patient ID(s) {sorted(timeline_ids)} do not match "
            f"note patient ID(s) {sorted(note_ids)}"
        )
    ensure_local_endpoint(args.base_url, args.allow_remote_endpoint)

    server_process: subprocess.Popen[bytes] | None = None
    server_log_handle: TextIO | None = None
    server_log = args.server_log or Path(str(args.output) + ".vllm.log")
    if not args.no_start_server:
        try:
            server_process, server_log_handle = start_vllm_server(
                args.base_url,
                args.model_path,
                args.model,
                args.tensor_parallel_size,
                args.startup_timeout,
                args.vllm_extra_args,
                server_log,
            )
        except (ValueError, RuntimeError, TimeoutError) as exc:
            parser.error(str(exc))
    elif not server_is_healthy(args.base_url):
        parser.error(f"no healthy model server at {args.base_url}")

    try:
        with open_text(args.note) as handle:
            note = handle.read()
        header, rows = read_table(args.timeline)
        targets = []
        for index, row in enumerate(rows):
            existing = current_span(row, note)
            if args.repair_all or existing is None or not spans_are_compatible(row.get("mention", ""), existing[2]):
                targets.append(index)

        audit_path = Path(str(args.output) + ".audit.jsonl")
        repaired = unresolved = 0
        audit_records: list[dict[str, Any]] = []
        for batch_number, indices in enumerate(chunks(targets, args.batch_size), start=1):
            batch_rows = [rows[index] for index in indices]
            print(f"batch {batch_number}: requesting {len(indices)} spans", file=sys.stderr)
            response = query_model(
                build_prompt(note, batch_rows), args.base_url, args.model,
                args.max_tokens, args.timeout, args.retries,
            )
            suggestions = extract_json_array(response)
            by_uid = {str(item.get("uid4", "")).strip(): item for item in suggestions}
            for index in indices:
                row = rows[index]
                uid4 = row.get("uid4", "").strip()
                suggestion = by_uid.get(uid4)
                resolved = resolve_suggestion(suggestion, note) if suggestion else None
                record: dict[str, Any] = {"uid4": uid4, "mention": row.get("mention", "")}
                if resolved is None:
                    row["char_pos"] = NULL_POSITION
                    row["char_pos_ub"] = NULL_POSITION
                    unresolved += 1
                    record.update({"status": "unresolved", "suggestion": suggestion})
                else:
                    start, end, selected = resolved
                    row["char_pos"] = str(start)
                    row["char_pos_ub"] = str(end)
                    repaired += 1
                    record.update({"status": "repaired", "start": start, "end": end, "text": selected})
                audit_records.append(record)

        write_table(args.output, header, rows)
        with audit_path.open("w", encoding="utf-8") as handle:
            for record in audit_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept = len(rows) - len(targets)
        print(
            f"wrote {args.output}: {repaired} repaired, {kept} kept, {unresolved} unresolved; "
            f"audit={audit_path}"
        )
        return 1 if unresolved else 0
    finally:
        if server_process is not None and not args.keep_server:
            print("stopping MiMo server", file=sys.stderr)
            stop_vllm_server(server_process, server_log_handle)


if __name__ == "__main__":
    sys.exit(main())
