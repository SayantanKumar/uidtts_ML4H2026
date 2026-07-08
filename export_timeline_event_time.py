#!/usr/bin/env python3
"""Export mention/time pairs from final timeline BSV files.

Input files are expected under:
    outputs/<model_name>/timelines/*_timelines_timeline_<1|2|3>.bsv[.gz]

For each source file, this script writes:
    outputs/<model_name>/timelines/timeline_<n>/<subject_id>.bsv
    outputs/<model_name>/timelines/timeline_<n>/<subject_id>.csv

The BSV output has no header and exactly two columns:
    mention|time

The CSV output has headers:
    event,time
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path
from typing import Iterable, TextIO


TIMELINE_FILE_RE = re.compile(
    r"^(?P<case_id>.+?)_timelines_timeline_(?P<timeline>[123])\.bsv(?:\.gz)?$"
)
SUBJECT_ID_RE = re.compile(r"(?<!\d)(\d{5,10})(?!\d)")
EXPECTED_COLUMNS = ["uid4", "mention", "time", "bounds", "known", "context_uid4s"]


def open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def subject_id_from_case_id(case_id: str) -> str:
    """Return the patient subject ID from either clean or decorated case names."""
    if case_id.isdigit():
        return case_id

    matches = SUBJECT_ID_RE.findall(case_id)
    if matches:
        return matches[-1]

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("_")
    if not safe:
        raise ValueError(f"Could not infer subject_id from case name: {case_id!r}")
    return safe


def assert_float(value: str, source: Path, row_num: int) -> str:
    try:
        float(value)
    except ValueError as exc:
        raise ValueError(
            f"{source}: row {row_num} has non-float time value {value!r}"
        ) from exc
    return value


def read_mention_time_rows(source: Path) -> list[tuple[str, str]]:
    with open_text(source) as handle:
        reader = csv.DictReader(handle, delimiter="|")
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"{source}: expected header {'|'.join(EXPECTED_COLUMNS)!r}, "
                f"found {'|'.join(reader.fieldnames or [])!r}"
            )

        rows: list[tuple[str, str]] = []
        for row_num, row in enumerate(reader, start=2):
            mention = (row.get("mention") or "").strip()
            time_value = (row.get("time") or "").strip()
            assert_float(time_value, source, row_num)
            rows.append((mention, time_value))

    return rows


def validate_bsv(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row_num, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"{path}: row {row_num} has {len(row)} columns")
            assert_float(row[1], path, row_num)


def validate_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["event", "time"]:
            raise ValueError(f"{path}: expected CSV header event,time")
        for row_num, row in enumerate(reader, start=2):
            if set(row) != {"event", "time"}:
                raise ValueError(f"{path}: row {row_num} does not have 2 columns")
            assert_float(row["time"], path, row_num)


def write_outputs(
    rows: list[tuple[str, str]],
    out_dir: Path,
    subject_id: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bsv_path = out_dir / f"{subject_id}.bsv"
    csv_path = out_dir / f"{subject_id}.csv"

    if not overwrite and (bsv_path.exists() or csv_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing output for {subject_id}: "
            f"{bsv_path} / {csv_path}. Use --overwrite to replace."
        )

    with bsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerows(rows)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["event", "time"])
        writer.writerows(rows)

    validate_bsv(bsv_path)
    validate_csv(csv_path)
    return bsv_path, csv_path


def iter_timeline_files(timelines_dir: Path) -> Iterable[tuple[Path, str, str]]:
    for source in sorted(timelines_dir.glob("*_timelines_timeline_*.bsv*")):
        match = TIMELINE_FILE_RE.match(source.name)
        if not match:
            continue
        timeline_id = match.group("timeline")
        subject_id = subject_id_from_case_id(match.group("case_id"))
        yield source, timeline_id, subject_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export mention/time pairs from final timeline BSV files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-name",
        help="Model output folder name under --outputs-root, e.g. qwen35.",
    )
    group.add_argument(
        "--timelines-dir",
        type=Path,
        help="Direct path to outputs/<model_name>/timelines.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root outputs directory. Used with --model-name. Default: outputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing timeline_<n>/<subject_id>.bsv and .csv files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any file fails. Otherwise failures are reported and processing continues.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timelines_dir = (
        args.timelines_dir
        if args.timelines_dir is not None
        else args.outputs_root / args.model_name / "timelines"
    )
    timelines_dir = timelines_dir.resolve()

    if not timelines_dir.is_dir():
        print(f"ERROR: timelines directory not found: {timelines_dir}", file=sys.stderr)
        return 2

    processed = 0
    failed = 0
    counts = {"1": 0, "2": 0, "3": 0}

    for source, timeline_id, subject_id in iter_timeline_files(timelines_dir):
        try:
            rows = read_mention_time_rows(source)
            out_dir = timelines_dir / f"timeline_{timeline_id}"
            bsv_path, csv_path = write_outputs(
                rows, out_dir, subject_id, overwrite=args.overwrite
            )
            processed += 1
            counts[timeline_id] += 1
            print(
                f"OK timeline_{timeline_id}: {source.name} -> "
                f"{bsv_path.name}, {csv_path.name} ({len(rows)} rows)"
            )
        except Exception as exc:  # noqa: BLE001 - report per-file failures cleanly.
            failed += 1
            print(f"ERROR: {source.name}: {exc}", file=sys.stderr)
            if args.strict:
                break

    print(
        "Summary: "
        f"processed={processed}, failed={failed}, "
        f"timeline_1={counts['1']}, timeline_2={counts['2']}, timeline_3={counts['3']}"
    )

    if processed == 0:
        print(
            f"ERROR: no timeline files found in {timelines_dir}",
            file=sys.stderr,
        )
        return 2

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())



#python export_timeline_event_time.py --timelines-dir outputs/gptoss120b/timelines --overwrite