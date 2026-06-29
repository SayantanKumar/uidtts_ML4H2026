# UID Anchor Detection Runbook

This runbook documents the current Biowulf workflow for
`uid_anchor_detection`, including the latest stepwise scripts, the combined
all-steps script, Hugging Face-only Step 5 embedding/reranking, selected-file
subset mode, and skip-existing rerun behavior.

The project goal is to align clinical events mentioned in free-text notes with
timestamped structured/tabular EHR rows.

```text
raw note text
  -> LLM-tagged event mentions
  -> uid4 mention table
  -> initial text-only temporal table
  -> structured-data anchor queries
  -> RAG retrieval over per-patient tabular data
  -> candidate timestamped evidence rows
  -> final revised timelines and character-position tables
```

## Current Recommended Scripts

Run scripts from:

```bash
/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection/run_scripts
```

| Use case | Script | Runs |
|---|---|---|
| Preferred full pipeline | `run_all_uid_pipeline.sh` | Steps 1-6 and post-processing in one job |
| Stepwise Steps 1-4 | `step_1234_tag_time.sh` | tagging, UID extraction, initial temporal extraction |
| Stepwise Step 5 | `step5_rag.sh` | UID bundle prep, anchor queries, HF RAG, reranking, source-row joining |
| Stepwise Step 6 + post | `step67.sh` | final timeline revision and post-processing |
| Debug Step 1-2 only | `step_12_uid_tagging.sh` | tagging and UID extraction |
| Debug Step 3-4 only | `step_34_temporal.sh` | initial temporal extraction |
| Debug Step 6 only | `step6_final_timeline.sh` | final timeline JSON generation |
| Debug post only | `post_processing.sh` | timeline/charpos extraction |

All scripts that support subset mode depend on:

```text
run_scripts/file_selection_utils.sh
```

The helper is sourced from:

```bash
$PROJECT_DIR/run_scripts/file_selection_utils.sh
```

This is important on Slurm because `sbatch` runs a copied script from
`/var/spool/slurm/...`; using `${BASH_SOURCE[0]}` alone will not reliably find
files beside the original submitted script.

The current workflow does not require `settings.sh` or `run_pipeline.sh`.

## Before Running

Edit hard-coded paths near the top of the relevant sbatch script before
submitting. The most important variables are:

```bash
PROJECT_DIR      # uid_anchor_detection project directory
INPUT_DIR        # input note folder for Step 1
OUTPUT_DIR       # output root, currently $PROJECT_DIR/outputs/qwen35/
TABULAR_DIR      # per-subject tabular CSV.GZ folder for Step 5
NOTE_LINK_PATH   # note_id,subject_id,hadm_id,admittime,dischtime mapping file
LLAMA_SERVER     # llama.cpp server binary
INSTRUCT_MODEL_PATH
INSTRUCT_MODEL_ALIAS
REASONING_MODEL_PATH
REASONING_MODEL_ALIAS
CUDA_DEVICES
```

Current scripts activate:

```bash
conda activate /data/kumars33/conda/envs/TTA
```

and load:

```bash
module load CUDA/12.8.1 gcc cmake
```

The scripts temporarily disable `set -u` during conda activation to avoid
Biowulf/conda deactivate-hook errors such as:

```text
xml_catalog_files_libxml2: unbound variable
```

## Running All Files Or A Selected Subset

Every current batch script supports two modes:

```bash
export FILE_SELECTION_MODE="all"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324.txt"
    # "11649157.txt"
)
```

Use all files:

```bash
export FILE_SELECTION_MODE="all"
```

Use only selected cases:

```bash
export FILE_SELECTION_MODE="subset"
SELECTED_INPUT_FILES=(
    "11121324.txt"
    "11649157.txt"
)
```

Bare case IDs are also accepted:

```bash
SELECTED_INPUT_FILES=(
    "11121324"
    "11649157"
)
```

For scripts that start from raw notes, subset mode stages only those `.txt`
files into a temporary selected-input folder. For later scripts, subset mode
filters/stages matching upstream outputs such as:

```text
tagged/<case>.txt
tags/<case>_tags.bsv.gz
temporal/<case>.txt.gz
temporal/answers/<case>_extracted.bsv.gz
bundle/<case>.txt
bundle/uidtts/comparison_results/<case>_joined.csv
```

For Step 5, subset mode also restricts tabular ingestion to:

```text
$TABULAR_DIR/<case>.csv.gz
```

## Inputs

### Notes

Step 1 reads note files from `INPUT_DIR`.

The current scripts are configured for plain text:

```bash
export INPUT_TYPE="txt"
export OUT_EXTENSION=".txt"
```

If notes are compressed, change these to:

```bash
export INPUT_TYPE="txt.gz"
export OUT_EXTENSION=".txt.gz"
```

### Tabular Data

Step 5 reads one compressed CSV per patient/subject:

```text
<subject_id>.csv.gz
```

Expected useful columns:

```text
t,event,value
```

The filename subject ID should match the case base used in upstream outputs:

```text
tagged/<subject_id>.txt
tags/<subject_id>_tags.bsv.gz
temporal/<subject_id>.txt.gz
temporal/answers/<subject_id>_extracted.bsv.gz
tabular/<subject_id>.csv.gz
```

### Note Link File

Step 5 and Step 6 use `NOTE_LINK_PATH`.

Expected columns:

```text
note_id,subject_id,hadm_id,admittime,dischtime
```

The Python scripts read `note_id` as a string, so numeric-looking IDs such as
`158331` are preserved correctly.

## Model Usage By Step

| Step | Model type |
|---|---|
| Step 1 | instruct LLM through llama.cpp |
| Step 2 | no LLM |
| Step 3 | reasoning LLM through llama.cpp |
| Step 4 | no LLM |
| Step 5A | no LLM |
| Step 5B | reasoning LLM through llama.cpp |
| Step 5C | local Hugging Face embedding model through `transformers` |
| Step 5D | local Hugging Face embedding + reranker models through `transformers` |
| Step 5E | no LLM |
| Step 6 | reasoning LLM through llama.cpp |
| Post-processing | no LLM |

The scripts keep separate instruct and reasoning model variables even when both
point to the same GGUF. That lets you switch them independently later.

Step 5 currently uses:

```bash
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"
HF_HOME="/data/kumars33/.cache/huggingface/"
```

These are loaded directly with Hugging Face `transformers`; no `sglang`, vLLM,
embedding server, or reranker server is used for Step 5C/5D.

Use a user-owned `HF_HOME`. Shared cache locations can fail with lock-file or
permission errors during model download.

## Submit The Pipeline

From Biowulf:

```bash
cd /data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection/run_scripts
mkdir -p sbatch_output sbatch_error
```

Preferred full pipeline:

```bash
sbatch run_all_uid_pipeline.sh
```

Stepwise pipeline:

```bash
sbatch step_1234_tag_time.sh
sbatch step5_rag.sh
sbatch step67.sh
```

Submit Step 6/7 only after Step 5 completes, or use a dependency:

```bash
jid5=$(sbatch --parsable step5_rag.sh)
sbatch --dependency=afterok:$jid5 step67.sh
```

Do not run `step67.sh` while Step 5 is still writing outputs. Step 6 depends on
complete `*_joined.csv` files from Step 5.

## Combined Script Behavior

`run_all_uid_pipeline.sh` is the current preferred script for full runs. It:

- starts one llama.cpp server for Steps 1, 3, and 5B
- stops that server before local HF Step 5C/5D
- starts a second llama.cpp server for Step 6
- logs start/end/duration and success/failed/skipped counts per section
- records per-case failures without stopping when the failure is case-specific
- skips existing non-empty or valid outputs where possible
- writes a summary TSV and failed-case TSV

Important combined-script logs:

```text
$OUTPUT_DIR/logs/all_pipeline.log
$OUTPUT_DIR/logs/all_pipeline_summary.tsv
$OUTPUT_DIR/logs/all_pipeline_failed_cases.tsv
```

## Step 1-4: Tagging And Initial Temporal Extraction

Batch scripts:

```text
run_scripts/run_all_uid_pipeline.sh
run_scripts/step_1234_tag_time.sh
```

Python scripts:

```text
prompt_io.py
extract_tagged_table.py
temporal_prompt_io.py
extract_temporal_table.py
```

### Step 1: Tag Clinical Mentions

Purpose:

Use the instruct LLM to mark clinical event mentions in each note with tags such
as `<tag_1>...</tag_1>`.

Inputs:

```text
$INPUT_DIR/*.txt or *.txt.gz
templates/tts_tag.txt
llama.cpp server on PORT=8080
```

Outputs:

```text
$OUTPUT_DIR/tagged/<case>.txt
```

Current long-note behavior:

```bash
STEP1_MAX_INPUT_LINES=500
STEP1_CHUNK_LINES=450
```

Files longer than `STEP1_MAX_INPUT_LINES` are chunked instead of skipped.
`prompt_io.py` renumbers tag IDs globally after chunking so downstream UID
extraction does not collide.

Important logs:

```text
$OUTPUT_DIR/logs/step1_prompt_io.log
$OUTPUT_DIR/logs/server_8080_step1.log
```

### Step 2: Extract Tags And Assign UID4 IDs

Purpose:

Parse the tagged notes and assign each mention a stable 4-character `uid4`.

Inputs:

```text
$OUTPUT_DIR/tagged/*.txt
```

Outputs:

```text
$OUTPUT_DIR/tags/<case>_tags.bsv.gz
```

Columns:

```text
tag_number|uid4|mention
```

Important log:

```text
$OUTPUT_DIR/logs/step2_extract_tags.log
```

### Step 3: Generate Initial Text-Only Temporal Tables

Purpose:

Use the reasoning LLM to estimate when each mention occurred relative to
admission.

Inputs:

```text
$OUTPUT_DIR/tagged/*.txt
$OUTPUT_DIR/tags/*_tags.bsv.gz
templates/temporal_prompt.txt
llama.cpp server on PORT=8080
```

Outputs:

```text
$OUTPUT_DIR/temporal/<case>.txt.gz
```

Expected content:

```text
uid4 | mention | time | bounds | known | context uid4s
```

Current long-case behavior:

```bash
STEP3_TAG_CHUNK_SIZE=60
```

`temporal_prompt_io.py` splits large mention tables into batches of up to this
many rows, queries the LLM per batch, and merges the rows into one temporal
output. This avoids the single giant Step 3 prompt that previously could hit
the llama.cpp context window and produce empty outputs.

Current empty-output behavior:

- existing Step 3 outputs are skipped only if they are readable and non-empty
- empty/corrupt/whitespace-only `.txt.gz` outputs are removed and regenerated
- empty LLM responses now raise an error instead of being written as success

Important logs:

```text
$OUTPUT_DIR/logs/step3_temporal_prompt_io.log
$OUTPUT_DIR/logs/server_8080_step3.log
```

### Step 4: Extract Temporal Tables

Purpose:

Parse the Step 3 LLM answer into compressed BSV tables.

Inputs:

```text
$OUTPUT_DIR/temporal/*.txt.gz
```

Outputs:

```text
$OUTPUT_DIR/temporal/answers/<case>_extracted.bsv.gz
```

Columns:

```text
uid4|mention|time|bounds|known|context uid4s
```

Important log:

```text
$OUTPUT_DIR/logs/step4_extract_temporal_table.log
```

## Step 5: Anchor Query Generation And RAG Matching

Batch scripts:

```text
run_scripts/run_all_uid_pipeline.sh
run_scripts/step5_rag.sh
```

Python scripts:

```text
mask_cr_append_mt.py
anchor_query_generator.py
rag_pipeline_llama_embed_folder_summarized_hf.py
anchor_match_ranker_hf.py
view_match_scores.py
join_anchor_with_source_timed.py
```

Step 5 has five substeps.

### Step 5A: Prepare UID-Tagged Bundle Reports

Purpose:

Convert `<tag_1>...</tag_1>` tags into `<uid4>...</uid4>` tags and append the
mention table to each report.

Inputs:

```text
$OUTPUT_DIR/tagged/*.txt
$OUTPUT_DIR/tags/*_tags.bsv.gz
```

Intermediate staging output:

```text
$OUTPUT_DIR/tagged_step5_gz/<case>.txt.gz
```

Outputs:

```text
$OUTPUT_DIR/bundle/<case>.txt
$OUTPUT_DIR/bundle/<case>_uid4.bsv
```

Important log:

```text
$OUTPUT_DIR/logs/step5a_mask_cr_append_mt.log
```

### Step 5B: Generate Anchor Queries

Purpose:

Ask the reasoning LLM what structured EHR evidence could help temporally anchor
each clinical event.

Inputs:

```text
$OUTPUT_DIR/bundle/*.txt
templates/aqg_system.txt
templates/aqg_uid.txt
llama.cpp server on PORT=8081 or the run-all shared PORT
```

Outputs:

```text
$OUTPUT_DIR/bundle/uidtts/<case>.json
```

Output shape:

```json
{
  "<uid4> event mention": [
    "candidate structured evidence description",
    "another possible evidence source"
  ]
}
```

Current long-report behavior:

```bash
AQG_CHUNK_SIZE=25
AQG_CONTEXT_WINDOW_CHARS=700
AQG_MAX_REPORT_CHARS=25000
AQG_MAX_QUERIES_PER_UID=3
```

`anchor_query_generator.py` generates anchor queries in UID batches and uses
compact case-report excerpts. If the model still fails to produce parseable
JSON for a UID batch, the script has fallback query generation so empty `{}` is
less likely.

Important logs:

```text
$OUTPUT_DIR/logs/step5b_anchor_query_generator.log
$OUTPUT_DIR/logs/server_8081.log
```

### Step 5C: Build RAG Indexes From Tabular Data

Purpose:

For each `<subject_id>.csv.gz`, summarize rows by event type, embed those
summaries with the local HF embedding model, and write a Chroma index.

Inputs:

```text
$TABULAR_DIR/<subject_id>.csv.gz
```

Outputs:

```text
$OUTPUT_DIR/bundle/tabular/uid_infix/uid_infix_<subject_id>.csv/
  chroma.sqlite3
  summary_mapping.json
```

The `summary_mapping.json` file is critical. It maps embedded summary chunks
back to original tabular row indices.

Important log:

```text
$OUTPUT_DIR/logs/step5c_rag_ingest.log
```

### Step 5D: Match Anchor Queries To Tabular Summaries

Purpose:

Retrieve and rerank tabular summaries that best match each anchor query.

Inputs:

```text
$OUTPUT_DIR/bundle/uidtts/*.json
$OUTPUT_DIR/bundle/tabular/uid_infix/uid_infix_<subject_id>.csv/
```

Outputs:

```text
$OUTPUT_DIR/bundle/uidtts/comparison_results/<case>_results.json
```

Important log:

```text
$OUTPUT_DIR/logs/step5d_anchor_ranker.log
```

### Step 5E: Flatten Scores And Join Back To Source Rows

Purpose:

Convert nested ranker JSON into CSVs, then join the matched summaries back to
concrete timestamped source rows from the patient tabular CSV.

Inputs:

```text
$OUTPUT_DIR/bundle/uidtts/comparison_results/<case>_results.json
$OUTPUT_DIR/bundle/tabular/uid_infix/uid_infix_<subject_id>.csv/summary_mapping.json
$TABULAR_DIR/<subject_id>.csv.gz
$NOTE_LINK_PATH
```

Outputs:

```text
$OUTPUT_DIR/bundle/uidtts/comparison_results/<case>_summary.csv
$OUTPUT_DIR/bundle/uidtts/comparison_results/<case>_joined.csv
```

The final `*_joined.csv` files are the key Step 5 outputs used by Step 6.

Useful columns:

```text
anchor
summary
score
bestscore
event
row_index
line_level_timestamp
line_level_event
line_level_value
hadm_id
cutoff_time
note_id
```

Important log:

```text
$OUTPUT_DIR/logs/step5e_flatten_join.log
```

## Step 6 + Post-Processing

Batch scripts:

```text
run_scripts/run_all_uid_pipeline.sh
run_scripts/step67.sh
```

Python scripts:

```text
assign_joint_timestamps.py
extract_timelines.py
get_tag_charpos_table.py
```

### Step 6: Final Joint Timestamp Revision

Purpose:

Use the reasoning LLM to revise the initial temporal table using:

- UID-tagged case report
- Step 4 initial mention table
- Step 5 joined tabular evidence
- admission/discharge context from `NOTE_LINK_PATH`

Inputs:

```text
$OUTPUT_DIR/bundle/<case>.txt
$OUTPUT_DIR/temporal/answers/<case>_extracted.bsv.gz
$OUTPUT_DIR/bundle/uidtts/comparison_results/<case>_joined.csv
$NOTE_LINK_PATH
templates/ajt.txt
llama.cpp server on PORT=8082 or the run-all Step 6 PORT
```

Outputs:

```text
$OUTPUT_DIR/bundle/uidtts/comparison_results/<case>_timelines.json
```

Current prompt-size controls:

```bash
STEP6_MAX_TOTAL_CANDIDATES=100
STEP6_TOP_K=3
STEP6_NUM_TIMELINES=3
STEP6_MAX_CANDIDATE_FIELD_CHARS=160
STEP6_MAX_CASE_REPORT_CHARS=30000
STEP6_CASE_CONTEXT_CHARS=220
```

The final JSON requests three alternate timelines per subject by default.
`assign_joint_timestamps.py` compacts the case-report context, truncates long
candidate fields, and stores retry/raw failure artifacts for failed cases.

Important logs:

```text
$OUTPUT_DIR/logs/step6_assign_joint_timestamps.log
$OUTPUT_DIR/logs/server_8082.log
$OUTPUT_DIR/logs/step6_retries/
$OUTPUT_DIR/logs/step6_failed_cases.txt
```

### Post-Processing: Extract Final Timelines

Purpose:

Convert Step 6 JSON outputs into final timeline BSV files.

Inputs:

```text
$OUTPUT_DIR/bundle/uidtts/comparison_results/*_timelines.json
```

Outputs:

```text
$OUTPUT_DIR/timelines/*.bsv.gz
```

Example names:

```text
158331_timelines_timeline_1.bsv.gz
158331_timelines_timeline_2.bsv.gz
158331_timelines_timeline_3.bsv.gz
```

### Post-Processing: Character Position Tables

Purpose:

Extract character spans for each UID-tagged mention and join those positions to
final and unrevised timelines.

Inputs:

```text
$OUTPUT_DIR/bundle/*.txt or *.txt.txt
$OUTPUT_DIR/timelines/*.bsv.gz
$OUTPUT_DIR/temporal/answers/*_extracted.bsv.gz
```

Outputs:

```text
$OUTPUT_DIR/bundle/charpos/*.bsv
$OUTPUT_DIR/bundle/charpos_unrevised/*.bsv
```

Important log:

```text
$OUTPUT_DIR/logs/post_processing.log
```

`post_processing.sh` and `step67.sh` safely handle both `*.txt` and `*.txt.txt`
bundle report names.

## Output Tree

For the current scripts:

```bash
OUT=/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection/outputs/qwen35
```

Expected final structure:

```text
$OUT/
  tagged/
    <subject_id>.txt
  tags/
    <subject_id>_tags.bsv.gz
  temporal/
    <subject_id>.txt.gz
    answers/
      <subject_id>_extracted.bsv.gz
  tagged_step5_gz/
    <subject_id>.txt.gz
  bundle/
    <subject_id>.txt
    <subject_id>_uid4.bsv
    uidtts/
      <subject_id>.json
      comparison_results/
        <subject_id>_results.json
        <subject_id>_summary.csv
        <subject_id>_joined.csv
        <subject_id>_timelines.json
    tabular/
      uid_infix/
        uid_infix_<subject_id>.csv/
          chroma.sqlite3
          summary_mapping.json
    charpos/
      *.bsv
    charpos_unrevised/
      *.bsv
  timelines/
    *.bsv.gz
  logs/
    all_pipeline.log
    all_pipeline_summary.tsv
    all_pipeline_failed_cases.tsv
    *.log
    step6_retries/
```

## Monitoring Progress

Set:

```bash
OUT=/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection/outputs/qwen35
```

### Monitor `run_all_uid_pipeline.sh`

```bash
tail -f $OUT/logs/all_pipeline.log
column -t -s $'\t' $OUT/logs/all_pipeline_summary.tsv
column -t -s $'\t' $OUT/logs/all_pipeline_failed_cases.tsv
```

### Monitor Step 1-4

```bash
watch -n 30 "
echo 'Step 1 tagged:'; find $OUT/tagged -maxdepth 1 \( -name '*.txt' -o -name '*.txt.gz' \) 2>/dev/null | wc -l
echo 'Step 2 tags:'; find $OUT/tags -maxdepth 1 -name '*_tags.bsv.gz' 2>/dev/null | wc -l
echo 'Step 3 temporal:'; find $OUT/temporal -maxdepth 1 \( -name '*.txt' -o -name '*.txt.gz' \) 2>/dev/null | wc -l
echo 'Step 4 extracted:'; find $OUT/temporal/answers -maxdepth 1 -name '*_extracted.bsv.gz' 2>/dev/null | wc -l
"
```

### Monitor Step 5

```bash
watch -n 30 "
echo '5A bundle reports:'; find $OUT/bundle -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l
echo '5B anchor jsons:'; find $OUT/bundle/uidtts -maxdepth 1 -name '*.json' 2>/dev/null | wc -l
echo '5C RAG mappings:'; find $OUT/bundle/tabular/uid_infix -name 'summary_mapping.json' 2>/dev/null | wc -l
echo '5D ranker results:'; find $OUT/bundle/uidtts/comparison_results -name '*_results.json' 2>/dev/null | wc -l
echo '5E joined csvs:'; find $OUT/bundle/uidtts/comparison_results -name '*_joined.csv' 2>/dev/null | wc -l
"
```

Useful logs:

```bash
tail -f $OUT/logs/step5b_anchor_query_generator.log
tail -f $OUT/logs/step5c_rag_ingest.log
tail -f $OUT/logs/step5d_anchor_ranker.log
tail -f $OUT/logs/step5e_flatten_join.log
```

### Monitor Step 6 + Post

```bash
watch -n 30 "
echo 'Step 6 timeline JSONs:'; find $OUT/bundle/uidtts/comparison_results -name '*_timelines.json' 2>/dev/null | wc -l
echo 'Final timeline BSVs:'; find $OUT/timelines -name '*.bsv.gz' 2>/dev/null | wc -l
echo 'Final charpos:'; find $OUT/bundle/charpos -name '*.bsv' 2>/dev/null | wc -l
echo 'Unrevised charpos:'; find $OUT/bundle/charpos_unrevised -name '*.bsv' 2>/dev/null | wc -l
"
```

Useful logs:

```bash
tail -f $OUT/logs/step6_assign_joint_timestamps.log
tail -f $OUT/logs/post_processing.log
tail -f $OUT/logs/server_8082.log
```

## Package Requirements

The current scripts need the usual scientific stack plus LangChain/OpenAI,
Chroma, and Hugging Face packages.

Core packages used in Step 5 onward:

```text
pandas
numpy
torch
transformers
scikit-learn
requests
httpx
openai
langchain-openai
langchain-core
langchain-chroma
chromadb
```

Older non-HF or graph variants may additionally need:

```text
langgraph
pydantic
```

## Copying Current Files To Biowulf

After local edits, copy updated Python scripts:

```bash
scp \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/prompt_io.py \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/temporal_prompt_io.py \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/anchor_query_generator.py \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/rag_pipeline_llama_embed_folder_summarized_hf.py \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/anchor_match_ranker_hf.py \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/assign_joint_timestamps.py \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/extract_timelines.py \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/get_tag_charpos_table.py \
  kumars33@helix.nih.gov:/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection/
```

Copy updated run scripts and helper:

```bash
scp \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/run_scripts/run_all_uid_pipeline.sh \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/run_scripts/step_1234_tag_time.sh \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/run_scripts/step5_rag.sh \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/run_scripts/step67.sh \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/run_scripts/file_selection_utils.sh \
  kumars33@helix.nih.gov:/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection/run_scripts/
```

Copy updated templates:

```bash
scp \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/templates/aqg_system.txt \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/templates/aqg_uid.txt \
  /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/uid_anchor_detection/templates/ajt.txt \
  kumars33@helix.nih.gov:/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection/templates/
```

## Common Failure Points

### `file_selection_utils.sh` not found under `/var/spool/slurm`

Cause:

`sbatch` runs a copied script from Slurm’s spool directory.

Fix:

Use the latest scripts, which source:

```bash
$PROJECT_DIR/run_scripts/file_selection_utils.sh
```

Also make sure `file_selection_utils.sh` was copied to Biowulf in
`$PROJECT_DIR/run_scripts/`.

### Empty Step 1/2 outputs

Check the llama-server log first:

```bash
tail -n 100 $OUT/logs/server_8080_step1.log
```

If you see CUDA out-of-memory, reduce model size, quantization, context, or GPU
load. Empty `tagged/` and `tags/` outputs usually mean the model/server failed
or returned unusable content.

### Empty Step 3 output

The latest `temporal_prompt_io.py` should no longer preserve empty outputs. It
removes empty/corrupt Step 3 files and regenerates them. If a case still fails,
inspect:

```bash
tail -n 200 $OUT/logs/step3_temporal_prompt_io.log
tail -n 200 $OUT/logs/server_8080_step3.log
```

For very long cases, reduce:

```bash
STEP3_TAG_CHUNK_SIZE=40
```

or smaller.

### `libcudart.so.12` missing

The llama.cpp binary was built against CUDA 12. Make sure the batch script loads
CUDA and includes CUDA libraries in `LD_LIBRARY_PATH`. The current scripts do:

```bash
module load CUDA/12.8.1 gcc cmake
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

### Missing `langchain_openai` or `chromadb`

Install missing packages into:

```bash
/data/kumars33/conda/envs/TTA
```

Typical missing Step 5 packages are:

```text
langchain-openai
langchain-chroma
chromadb
```

### Hugging Face cache permission error

Use a user-owned cache:

```bash
export HF_HOME="/data/kumars33/.cache/huggingface/"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
```

Avoid writing model lock files into a shared cache directory that you do not
own.

### Step 5 cannot find tabular files

Step 5 derives `patient_id` from the result filename:

```bash
patient_id="${base_name%%-*}"
```

For the current workflow, keep filenames subject-ID-only before Step 5:

```text
tagged/<subject_id>.txt
tags/<subject_id>_tags.bsv.gz
temporal/<subject_id>.txt.gz
temporal/answers/<subject_id>_extracted.bsv.gz
tabular/<subject_id>.csv.gz
```

### Step 5 complete but Step 6 has no inputs

Step 6 requires:

```text
$OUT/bundle/uidtts/comparison_results/*_joined.csv
```

If only `*_results.json` files exist, Step 5D finished but Step 5E did not.
Check:

```bash
tail -n 100 $OUT/logs/step5e_flatten_join.log
```

### Step 6 JSON exists but post-processing has no BSVs

Check:

```bash
tail -n 100 $OUT/logs/post_processing.log
```

The final timeline JSON must contain a valid `timelines` field with non-empty
timeline `data`.

## Clean Reruns

The current scripts are designed for additive reruns.

Existing complete outputs are skipped:

- Step 1 skips existing non-empty tagged notes.
- Step 2 skips existing non-empty tag tables.
- Step 3 skips only readable non-empty temporal outputs.
- Step 4 skips existing non-empty extracted temporal tables.
- Step 5B skips existing valid anchor-query JSONs.
- Step 5C leaves existing RAG outputs in place.
- Step 5E skips existing non-empty `*_joined.csv`.
- Step 6 skips existing valid `*_timelines.json`.
- Post-processing skips existing non-empty final BSV outputs.

Missing, empty, corrupt, or invalid outputs are regenerated where the script can
detect them.

For partial reruns, the safest approach is subset mode:

```bash
export FILE_SELECTION_MODE="subset"
SELECTED_INPUT_FILES=(
    "11121324.txt"
)
```

Avoid deleting shared upstream folders unless you are intentionally rerunning
the entire pipeline.
