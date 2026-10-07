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
| Standard | Full anchor-query retrieval and UID-aware final revision | `submit.sh` | `outputs/<output-name>/timelines/` |
| Blind | Uses standard retrieval, but hides matched row timestamps/events/values and admission/discharge context from the Step 6 reasoner | `submit_blind.sh` | `outputs/<output-name>/timelines_blind/` |
| E2RR | Replaces anchor-query matching in Steps 5D-5E with direct event-to-row retrieval and reranking | `submit_e2rr.sh` | `outputs/<output-name>/timelines_e2rr/` |
| No UID | Uses standard retrieval, but removes UID fields and tags from the Step 6 model input; a fail-closed audit checks for UID leakage | `submit_nouid.sh` | `outputs/<output-name>/timelines_nouid/` |
| No Reranker | Keeps anchor-query retrieval, but ranks candidates only by embedding cosine similarity and skips the reranker model | `submit_noranker.sh` | `outputs/<output-name>/timelines_noranker/` |

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

The `run_scripts_slurm/` scripts are Slurm/Biowulf scripts. The submission
wrappers define the source-data and GGUF model paths used by every stage. Other
environment-specific settings remain in the stage configuration blocks,
especially:

- `PROJECT_DIR`
- llama.cpp server path
- Hugging Face model/cache paths
- Slurm resources and email address

The wrappers submit each stage with `afterok` dependencies. Run one from the
repository root. With no model arguments, they retain the Qwen 3.5 defaults:

```bash
bash submit.sh
```

Run the wrappers with `bash`, not `sbatch`: each wrapper is a submission
controller that calls `sbatch` for its own stages. It prints the model
configuration first and the complete dependency chain after all stage jobs
have been submitted.

### Source data paths

Every wrapper currently specifies the same three source paths:

```text
notes:      /data/grimalditersans/mimic_uid_data/i2m4_n40_notes/
structured: /data/grimalditersans/mimic_uid_data/i2m4_n40_filtered_time/
note link:  /data/grimalditersans/mimic_uid_data/note_link_i2m4.csv
```

Edit `WRAPPER_NOTES_DIR`, `WRAPPER_STRUCTURED_DIR`, and `WRAPPER_NOTE_LINK` in
the wrappers to change their hard-coded defaults. For a one-off run, override
them on the command line:

```bash
bash submit.sh \
  --notes-dir /data/my_dataset/notes/ \
  --structured-dir /data/my_dataset/tabular/ \
  --note-link /data/my_dataset/note_link.csv
```

The wrapper exports these values to every applicable Slurm stage. Related
ablation runs should use the same source paths as their prerequisite standard
run.

### LLM GPU resources

Each wrapper defines the GPU type and count used by llama-server jobs:

```bash
WRAPPER_GPU_TYPE="a100"
WRAPPER_GPU_COUNT=4
```

These values override the corresponding `#SBATCH --gres` directive for Steps
1-2, 3-4, 5B, and 6. The CUDA device list is derived automatically, so a count
of four produces `CUDA_DEVICES=0,1,2,3`. Embedding and reranking stages retain
their smaller stage-specific GPU requests.

For a one-off resource override, use:

```bash
bash submit.sh --gpu-type a100 --gpu-count 4
```

### Model aliases and output paths

The model alias has two roles: it is the model name sent to llama-server, and
by default it is also the output-folder name. For example:

```text
model alias: qwen35_2
output root: $PROJECT_DIR/outputs/qwen35_2/
```

The standard and ablation wrappers must resolve to the same output root so the
ablations can reuse prerequisite artifacts. Therefore, use the same alias for
the full pipeline and every related ablation. Ablations keep the shared root
and write to separate subfolders such as `timelines_blind/`, `timelines_e2rr/`,
`timelines_nouid/`, and `timelines_noranker/`; they do not change the root
folder name.

To select a different GGUF model, pass its path and a short alias. The alias is
also used for the output folder, so this example writes beneath
`outputs/glm52/`:

```bash
bash submit.sh \
  --model-path /data/models/GLM-5.2-00001-of-00006.gguf \
  --model-alias glm52
```

The instruct and reasoning stages can use different models. When their aliases
differ, the default output name is `<instruct-alias>-to-<reasoning-alias>`:

```bash
bash submit.sh \
  --instruct-model-path /data/models/instruct/model.gguf \
  --instruct-model-alias instruct7b \
  --reasoning-model-path /data/models/reasoning/model.gguf \
  --reasoning-model-alias reasoner70b
```

That example writes beneath `outputs/instruct7b-to-reasoner70b/`. Use
`--output-name NAME` to choose another folder under `PROJECT_DIR/outputs`, or
`--output-dir DIR` to provide the complete output path. Run any wrapper with
`--help` for the complete option list. If either output override is used, pass
the same override to all related ablation wrappers.

For example, this standard run and all four ablations share
`outputs/qwen35_2/`:

```bash
bash submit.sh --model-alias qwen35_2
bash submit_blind.sh --model-alias qwen35_2
bash submit_e2rr.sh --model-alias qwen35_2
bash submit_nouid.sh --model-alias qwen35_2
bash submit_noranker.sh --model-alias qwen35_2
```

The standard wrapper runs Steps 1-6 and post-processing. E2RR and No Reranker
start at Step 5D and therefore require Steps 1-5C for the same output folder.
Blind and No UID start at Step 6 and require the standard results through Step
5E. Pass the same model and output arguments to the prerequisite and ablation
wrappers so they read and write the same model-specific folder.

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

Unless overridden with `--output-name` or `--output-dir`, artifacts are written
beneath `outputs/<model-alias>/`. The hard-coded default remains
`outputs/qwen35_2/`:

```text
outputs/<output-name>/
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

### Standalone character-position post-processing

To create or repair character-position tables without rerunning the pipeline:

```bash
python postprocess_char_positions.py outputs/glm5.2
```

The script reads tagged reports and `*_uid4.bsv` mappings from `bundle/`, then
writes tables to `bundle/charpos/`. It recognizes both `<uid4>` and `<tag_N>`
markup, validates `tags/*_tags.bsv.gz` tag-to-UID mappings against the bundle's
declared mentions (to handle dropped or renumbered tags), replaces header-only
outputs automatically, and joins a `timelines/` or
`timelines_final/` directory when one exists. To choose paths explicitly:

```bash
python postprocess_char_positions.py outputs/glm5.2 \
  --source-dir /path/to/original/reports \
  --timeline-dir outputs/glm5.2/timelines_final \
  --output-dir outputs/glm5.2/bundle/charpos \
  --overwrite --strict
```

Use `--source-dir` whenever offsets must refer to the original input reports.
Without it, offsets refer to the model-generated bundle text after tags are
removed, whose headings or whitespace may differ from the source documents.

### LLM-assisted character-position repair

`repair_char_positions_llm.py` can repair a position table against one original
note with MiMo through vLLM. It starts and stops the local model server
automatically unless `--no-start-server` is passed. The model selects exact
quotes; the script computes and validates the offsets and preserves all other
BSV columns. Inside a GPU allocation, repair a table to a new file with:

```bash
python repair_char_positions_llm.py \
  --timeline /data/grimalditersans/MLHC_TTA/compare_tts/m4rnd20_batch_output_0001/bundle/charpos/11649157-DS-7_positions_timeline_1.bsv \
  --note /data/grimalditersans/mimic_uid_data/i2m4_n40_notes/2023-03-31_10-31-14_annotations_162755_yes_structured.txt \
  --output /data/grimalditersans/MLHC_TTA/compare_tts/m4rnd20_batch_output_0001/bundle/charpos_llm/11649157-DS-7_positions_timeline_1.bsv \
  --model-path /data/grimalditersans/models/MiMo-V2.6-Pro-RL/ \
  --tensor-parallel-size 1
```

The timeline and note examples above identify different cases; substitute a
note with the same patient ID as the timeline in an actual run. Add
`--repair-all` to review every row instead of only missing or implausible spans.
An adjacent `.audit.jsonl` records every accepted or unresolved repair.

To request one Biowulf `gpuh200` node with all eight GPUs and run the example
end to end, submit the included wrapper:

```bash
sbatch run_scripts_slurm/repair_char_positions_mimo_h200.sh
```

Set `TIMELINE_FILE`, `NOTE_FILE`, and `OUTPUT_FILE` before `sbatch` to process a
different matching patient without editing the wrapper.

## Script Layout

- `run_scripts_slurm/`: current stepwise pipeline and ablation scripts
- `run_scripts_slurm/models/`: model-specific full and partial pipeline scripts
- `run_scripts_slurm/parallel/`: parallel/batch versions of the main stages
- `templates/`: tagging, temporal, anchor-query, and final-revision prompts
- `run_scripts/`: earlier pipeline scripts retained for compatibility
- `UID_ANCHOR_DETECTION_RUNBOOK.md`: detailed operational background for the
  earlier `run_scripts/` workflow
