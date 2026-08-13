# UID Anchor Detection

This repository aligns clinical events mentioned in free-text notes with
timestamped rows from structured EHR data. It also includes ablations for
measuring the effects of anchor-query generation, timestamp visibility, and
stable mention identifiers (`uid4`), as well as retrieval reranking.

## Pipeline

```text
clinical note
  -> tag event mentions and assign uid4 identifiers       (Steps 1-2)
  -> create an initial text-only temporal table           (Steps 3-4)
  -> build note/mention bundles                           (Step 5A)
  -> generate structured-evidence anchor queries          (Step 5B)
  -> embed patient-specific tabular rows                  (Step 5C)
  -> retrieve, rerank, and join timestamped evidence      (Steps 5D-5E)
  -> revise the timeline with an LLM                      (Step 6)
  -> export timeline and character-position tables       (post-processing)
```

The standard pipeline uses generated anchor queries to retrieve relevant EHR
rows. The final reasoner receives the note, initial mention table, matched
structured evidence, and admission/discharge context.

## Ablations

| Variant | What changes | Submission wrapper | Final timeline directory |
|---|---|---|---|
| Standard | Full anchor-query retrieval and UID-aware final revision | `submit_qwen.sh` | `outputs/qwen35/timelines/` |
| Blind | Uses standard retrieval, but hides matched row timestamps/events/values and admission/discharge context from the Step 6 reasoner | `submit_qwen_blind.sh` | `outputs/qwen35/timelines_blind/` |
| E2RR | Replaces anchor-query matching in Steps 5D-5E with direct event-to-row retrieval and reranking | `submit_qwen_e2rr.sh` | `outputs/qwen35/timelines_e2rr/` |
| No UID | Uses standard retrieval, but removes UID fields and tags from the Step 6 model input; a fail-closed audit checks for UID leakage | `submit_qwen_nouid.sh` | `outputs/qwen35/timelines_nouid/` |
| No Reranker | Keeps anchor-query retrieval, but ranks candidates only by embedding cosine similarity and skips the reranker model | `submit_qwen_noranker.sh` | `outputs/qwen35/timelines_noranker/` |

E2RR uses `row_match_ranker_hf.py` and `join_row_with_source_timed.py`. The
No Reranker variant uses `anchor_match_noranker_hf.py`, followed by the standard
anchor/source join and final timeline reasoner. The other variants reuse the
standard Step 5 results and change the final timeline script and prompt:

| Variant | Step 6 implementation | Prompt |
|---|---|---|
| Standard | `assign_joint_timestamps.py` | `templates/ajt.txt` |
| Blind | `assign_joint_timestamp_blind.py` | `templates/ajt_blind.txt` |
| E2RR | `assign_joint_timestamp_noquery.py` | `templates/ajt.txt` |
| No UID | `assign_joint_timestamp_nouid.py` | `templates/ajt_nouid.txt` |
| No Reranker | `assign_joint_timestamps.py` | `templates/ajt.txt` |

## Running on Biowulf

The `run_scripts_qwen/` scripts are Slurm/Biowulf scripts. Before submitting,
edit their configuration blocks to match your environment, especially:

- `PROJECT_DIR`, `INPUT_DIR`, `OUTPUT_DIR`, and `TABULAR_DIR`
- `NOTE_LINK_PATH`
- llama.cpp server and GGUF model paths
- Hugging Face model/cache paths
- Slurm resources and email address

The wrappers submit each stage with `afterok` dependencies. Run one from the
repository root, for example:

```bash
bash submit_qwen.sh
```

To run an ablation instead:

```bash
bash submit_qwen_blind.sh
bash submit_qwen_e2rr.sh
bash submit_qwen_nouid.sh
bash submit_qwen_noranker.sh
```

Each wrapper runs Steps 1-4, Steps 5A-5E, Step 6, and the matching
post-processing script. The E2RR wrapper selects `step_5d_e2rr.sh` and
`step_5e_e2rr.sh`; blind and no-UID select their own Step 6 and post-processing
scripts; No Reranker selects its own Steps 5D-6 and post-processing scripts.

For a single case, set the same selection in every script used by the wrapper:

```bash
export FILE_SELECTION_MODE="subset"
SELECTED_INPUT_FILES=(
    "11121324.txt"
)
```

Use `FILE_SELECTION_MODE="all"` to process every available case. Existing
valid outputs are generally skipped, allowing interrupted jobs to resume.

## Inputs

The pipeline expects:

- clinical notes as `.txt` or `.txt.gz` files
- one patient-specific structured EHR file per subject, normally
  `<subject_id>.csv.gz`, with timestamp, event, and value data
- a note-link file containing identifiers and hospitalization times, including
  `note_id`, `subject_id`, `hadm_id`, `admittime`, and `dischtime`
- local LLM, embedding, and reranking models

## Main Outputs

With the default `OUTPUT_DIR=outputs/qwen35`, intermediate and final artifacts
are organized as follows:

```text
outputs/qwen35/
  tagged/                                  tagged notes
  tags/                                    uid4 mention tables
  temporal/answers/                        initial temporal tables
  bundle/uidtts/comparison_results/        standard joined evidence and JSON
  bundle/uidtts/e2rr_comparison_results/   E2RR joined evidence and JSON
  bundle/uidtts/blind_comparison_results/  blind Step 6 JSON
  bundle/uidtts/nouid_comparison_results/  no-UID Step 6 JSON
  bundle/uidtts/norerank_comparison_results/ no-reranker joined evidence and JSON
  timelines/                               standard final timeline tables
  timelines_blind/                         blind final timeline tables
  timelines_e2rr/                          E2RR final timeline tables
  timelines_nouid/                         no-UID final timeline tables
  timelines_noranker/                      no-reranker final timeline tables
  logs/                                    server, stage, retry, and model-input logs
```

Final timeline tables are compressed BSV files. Post-processing also creates
character-position tables that link timeline events back to mention spans.
Treat notes, model-input logs, joined evidence, and outputs as protected
clinical data.

## Script Layout

- `run_scripts_qwen/`: current stepwise pipeline and ablation scripts
- `run_scripts_qwen/models/`: model-specific full and partial pipeline scripts
- `run_scripts_qwen/parallel/`: parallel/batch versions of the main stages
- `templates/`: tagging, temporal, anchor-query, and final-revision prompts
- `run_scripts/`: earlier pipeline scripts retained for compatibility
- `UID_ANCHOR_DETECTION_RUNBOOK.md`: detailed operational background for the
  earlier `run_scripts/` workflow
