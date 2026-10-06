#!/usr/bin/env python3
"""Create character-position tables from tagged case reports.

This is a standalone, standard-library-only post-processing step.  It supports
both bundle formats used by this repository:

* ``<02d2>mention</02d2>`` (UID tags)
* ``<tag_1 ...>mention</tag_1>`` (numbered tags)

Numbered tags are mapped to UID4 values by row order in the matching
``*_uid4.bsv`` file.  Offsets refer to the report after markup is removed and
use Python's usual half-open convention: ``text[char_pos:char_pos_ub]``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


TAG_RE = re.compile(
    r"<(?P<tag>[0-9a-fA-F]{4}|tag_(?P<number>\d+))(?:\s[^>]*)?>"
    r"(?P<content>.*?)</(?P=tag)\s*>",
    re.DOTALL | re.IGNORECASE,
)
SUBJECT_ID_RE = re.compile(r"(?<!\d)\d{5,}(?!\d)")


@dataclass(frozen=True)
class Span:
    uid4: str
    start: int
    end: int
    mention: str


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def case_base(path: Path) -> str:
    name = path.name
    while True:
        stem = Path(name).stem
        if stem == name:
            break
        name = stem
    for suffix in (
        "_timeline_extracted", "_results", "_summary", "_positions_1",
        "_positions", "_timelines", "_timeline", "_extracted",
        "_joined", "_matches", "_tags", "_modified", "_uid4",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def identifiers(name: str) -> list[str]:
    base = case_base(Path(name))
    result = [base] if base else []
    for value in SUBJECT_ID_RE.findall(base):
        if value not in result:
            result.append(value)
    return result


def name_matches(path: Path, values: Iterable[str]) -> bool:
    candidate = case_base(path)
    if candidate in values:
        return True
    return any(
        re.search(rf"(^|[^0-9]){re.escape(value)}([^0-9]|$)", path.name)
        for value in values
    )


def find_related(directory: Path, report: Path, patterns: Iterable[str]) -> list[Path]:
    values = identifiers(report.name)
    report_base = case_base(report)
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(p for p in directory.glob(pattern) if name_matches(p, values))
    return sorted(set(matches), key=lambda path: (case_base(path) != report_base, path.name))


def load_uid_table(path: Path | None) -> tuple[list[str], dict[str, str]]:
    ordered: list[str] = []
    mentions: dict[str, str] = {}
    if path is None:
        return ordered, mentions
    with open_text(path) as handle:
        for row in csv.reader(handle, delimiter="|"):
            if len(row) < 2:
                continue
            uid4 = row[0].strip().lower()
            if not re.fullmatch(r"[0-9a-f]{4}", uid4):
                continue
            ordered.append(uid4)
            mentions[uid4] = row[1].strip()
    return ordered, mentions


def extract_spans(text: str, numbered_uids: list[str]) -> tuple[str, list[Span]]:
    """Remove paired tags and return spans in the resulting plain text."""
    clean_parts: list[str] = []
    spans: list[Span] = []
    source_cursor = 0
    clean_cursor = 0

    for match in TAG_RE.finditer(text):
        prefix = text[source_cursor:match.start()]
        clean_parts.append(prefix)
        clean_cursor += len(prefix)

        content = match.group("content")
        # LLM tags commonly contain padding spaces outside the actual mention.
        left_trim = len(content) - len(content.lstrip())
        right_trimmed = content.rstrip()
        start = clean_cursor + left_trim
        end = clean_cursor + len(right_trimmed)

        number = match.group("number")
        if number is None:
            uid4 = match.group("tag").lower()
        else:
            index = int(number) - 1
            uid4 = numbered_uids[index] if 0 <= index < len(numbered_uids) else f"tag_{number}"

        clean_parts.append(content)
        clean_cursor += len(content)
        spans.append(Span(uid4, start, end, content.strip()))
        source_cursor = match.end()

    clean_parts.append(text[source_cursor:])
    return "".join(clean_parts), spans


def flexible_occurrences(text: str, mention: str) -> list[tuple[int, int]]:
    """Find a mention case-insensitively while allowing whitespace variation."""
    mention = mention.strip()
    if not mention:
        return []
    pattern = r"\s+".join(re.escape(part) for part in re.split(r"\s+", mention))
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]


def align_spans_to_source(spans: list[Span], source_text: str) -> tuple[list[Span], int]:
    """Relocate generated tagged spans into the original source report."""
    aligned: list[Span] = []
    used: set[tuple[int, int]] = set()
    cursor = 0
    missing = 0
    for span in spans:
        candidates = flexible_occurrences(source_text, span.mention)
        available = [candidate for candidate in candidates if candidate not in used]
        choice = next((candidate for candidate in available if candidate[0] >= cursor), None)
        if choice is None and available:
            choice = available[0]
        if choice is None:
            missing += 1
            continue
        used.add(choice)
        cursor = max(cursor, choice[1])
        aligned.append(Span(span.uid4, choice[0], choice[1], source_text[choice[0]:choice[1]]))
    return aligned, missing


def load_timeline(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open_text(path) as handle:
        rows = list(csv.reader(handle, delimiter="|"))
    if not rows:
        return [], []
    default = ["uid4", "mention", "time", "bounds", "known", "context_uid4s"]
    first = [value.strip() for value in rows[0]]
    has_header = bool(first and first[0].lower() == "uid4")
    header = first if has_header else default[: len(first)]
    data: list[dict[str, str]] = []
    for row in rows[1:] if has_header else rows:
        if not row or row[0].strip().lower() == "uid4":
            continue
        padded = row + [""] * (len(header) - len(row))
        data.append(dict(zip(header, padded)))
    return header, data


def choose_span(
    uid4: str,
    mention: str,
    tagged: dict[str, list[Span]],
    text: str,
    used: set[tuple[int, int]],
) -> tuple[int | None, int | None]:
    for span in tagged.get(uid4.lower(), []):
        key = (span.start, span.end)
        if key not in used:
            used.add(key)
            return key
    candidates = flexible_occurrences(text, mention)
    for candidate in candidates:
        if candidate not in used:
            used.add(candidate)
            return candidate
    return None, None


def has_position_rows(path: Path) -> bool:
    """Return True only when an existing table contains a populated span."""
    try:
        with open_text(path) as handle:
            reader = csv.reader(handle, delimiter="|")
            header = [column.strip().lower() for column in next(reader, [])]
            start_index = header.index("char_pos")
            end_index = header.index("char_pos_ub")
            for row in reader:
                if (
                    len(row) > max(start_index, end_index)
                    and row[start_index].strip().isdigit()
                    and row[end_index].strip().isdigit()
                ):
                    return True
    except (OSError, UnicodeError, ValueError):
        pass
    return False


def write_raw(path: Path, spans: list[Span], mention_map: dict[str, str]) -> tuple[int, int]:
    resolved = [span for span in spans if re.fullmatch(r"[0-9a-f]{4}", span.uid4)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerow(["uid4", "char_pos", "char_pos_ub", "mention"])
        for span in resolved:
            writer.writerow([
                span.uid4, span.start, span.end,
                mention_map.get(span.uid4, span.mention),
            ])
    return len(resolved), len(spans) - len(resolved)


def write_joined(path: Path, timeline: Path, text: str, spans: list[Span]) -> tuple[int, int]:
    header, rows = load_timeline(timeline)
    normalized = {column.strip().lower(): column for column in header}
    uid_key = normalized.get("uid4", header[0])
    mention_key = normalized.get("mention", "mention")
    output_header = list(header)
    insert_at = output_header.index(uid_key) + 1
    if "char_pos" not in normalized:
        output_header.insert(insert_at, "char_pos")
        insert_at += 1
    if "char_pos_ub" not in normalized:
        output_header.insert(insert_at, "char_pos_ub")

    tagged: dict[str, list[Span]] = {}
    for span in spans:
        tagged.setdefault(span.uid4.lower(), []).append(span)
    used: set[tuple[int, int]] = set()
    found = 0

    with open_output(path) as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerow(output_header)
        for row in rows:
            uid4 = row.get(uid_key, "").strip()
            start, end = choose_span(uid4, row.get(mention_key, ""), tagged, text, used)
            found += start is not None
            row[normalized.get("char_pos", "char_pos")] = "" if start is None else str(start)
            row[normalized.get("char_pos_ub", "char_pos_ub")] = "" if end is None else str(end)
            writer.writerow([row.get(column, "") for column in output_header])
    return found, len(rows) - found


def output_name(report: Path, timeline: Path | None) -> str:
    base = case_base(report)
    if timeline is None:
        return f"{base}_positions.bsv"
    return timeline.name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find mention character positions in tagged bundle reports."
    )
    parser.add_argument("output_root", type=Path, help="Pipeline output root (for example outputs/glm5.2)")
    parser.add_argument("--bundle-dir", type=Path, help="Tagged reports and *_uid4.bsv files")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Original TXT/TXT.GZ reports whose character coordinate system should be used",
    )
    parser.add_argument("--timeline-dir", type=Path, help="Optional BSV/BSV.GZ timelines to join")
    parser.add_argument("--output-dir", type=Path, help="Destination (default: BUNDLE/charpos)")
    parser.add_argument("--pattern", default="*.txt*", help="Bundle report glob (default: *.txt*)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing position tables")
    parser.add_argument("--strict", action="store_true", help="Fail if any timeline position remains empty")
    args = parser.parse_args()

    root = args.output_root.resolve()
    bundle = (args.bundle_dir or root / "bundle").resolve()
    destination = (args.output_dir or bundle / "charpos").resolve()
    timeline_dir = args.timeline_dir.resolve() if args.timeline_dir else None
    source_dir = args.source_dir.resolve() if args.source_dir else None
    if timeline_dir is None:
        for candidate in (root / "timelines", root / "timelines_final"):
            if candidate.is_dir() and any(candidate.glob("*.bsv*")):
                timeline_dir = candidate.resolve()
                break

    if not bundle.is_dir():
        parser.error(f"bundle directory does not exist: {bundle}")
    if source_dir is not None and not source_dir.is_dir():
        parser.error(f"source directory does not exist: {source_dir}")
    destination.mkdir(parents=True, exist_ok=True)

    reports = [
        path for path in sorted(bundle.glob(args.pattern))
        if path.is_file() and not path.name.endswith("_uid4.bsv")
    ]
    if not reports:
        parser.error(f"no reports matching {args.pattern!r} in {bundle}")

    files_written = found_total = missing_total = source_errors = 0
    for report in reports:
        uid_files = find_related(bundle, report, ("*_uid4.bsv", "*_uid4.bsv.gz"))
        uid_file = uid_files[0] if uid_files else None
        ordered_uids, mention_map = load_uid_table(uid_file)
        with open_text(report) as handle:
            clean_text, spans = extract_spans(handle.read(), ordered_uids)

        coordinate_text = clean_text
        alignment_missing = 0
        if source_dir is not None:
            source_files = find_related(source_dir, report, ("*.txt", "*.txt.gz"))
            if not source_files:
                print(f"ERROR: no original source report found for {report.name}", file=sys.stderr)
                source_errors += 1
                continue
            with open_text(source_files[0]) as handle:
                coordinate_text = handle.read()
            spans, alignment_missing = align_spans_to_source(spans, coordinate_text)
            if alignment_missing:
                print(
                    f"WARNING: {report.name}: {alignment_missing} tagged mentions "
                    "could not be aligned to the original report",
                    file=sys.stderr,
                )

        timelines = (
            find_related(timeline_dir, report, ("*.bsv", "*.bsv.gz"))
            if timeline_dir and timeline_dir.is_dir() else []
        )
        jobs: list[Path | None] = timelines or [None]
        for timeline in jobs:
            target = destination / output_name(report, timeline)
            # Header-only files are incomplete and should be repaired by default.
            if has_position_rows(target) and not args.overwrite:
                print(f"skip existing: {target}")
                continue
            found, missing = (
                write_joined(target, timeline, coordinate_text, spans)
                if timeline else write_raw(target, spans, mention_map)
            )
            if timeline is None:
                missing += alignment_missing
            print(f"wrote {target} ({found} positioned, {missing} missing)")
            files_written += 1
            found_total += found
            missing_total += missing

    print(f"done: {files_written} files, {found_total} positioned rows, {missing_total} missing rows")
    if source_errors or (args.strict and missing_total):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
