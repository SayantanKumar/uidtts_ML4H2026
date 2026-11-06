# UID-preserving clinical timeline reconstruction

Code for **Anchoring Clinical Events in Time: UID-Preserving Multimodal Reconstruction and Source-Grounded Adjudication** to appear in the **Proceedings of Machine Learning in Health Symposium (ML4H 2026)**. The pipeline connects clinical event occurrences in narrative notes to timestamped structured EHR evidence. A four-character mention identifier (`uid4`) links each occurrence across text-only temporal estimation, retrieval, source-row grounding, and joint timeline revision. The bundled GAVEL tools compare pairs of timelines against their source evidence.

The Python scripts and `templates/` directory are at this repository's root. There is no additional `uid_anchor_detection/` subdirectory to enter.

## Data access and publication

Obtain MIMIC data through the applicable [PhysioNet credentialing and data-use process](https://physionet.org/content/mimiciv/3.1/). MIMIC is deidentified but remains access-controlled under the [PhysioNet data license](https://physionet.org/content/mimiciv/view-license/3.1/). Dataset access does not grant permission to publish patient-level records. Keep notes, encounter identifiers, linkage tables, model prompts and responses, retrieved evidence, vector stores, logs, and review packets in an access-controlled location outside the Git checkout. These scripts transform clinical information; they do **not** de-identify it. In particular, `mask_cr_append_mt.py` replaces tag numbers with mention UIDs, not identifying information with redactions.

The source snapshot includes historical examples and site-specific settings that require review before public release. Do not assume a file is safe to publish because it is named `example`, is a comment, or contains de-identified dataset identifiers. Use independently authored synthetic material for public demonstrations.

## Workflow and script reference

```text
note -> numbered event tags -> uid4 mention table
     -> initial text-only temporal table
     -> UID-tagged note and anchor queries
structured rows -> event-series summaries and embeddings
anchor queries + embeddings -> retrieved/reranked summaries
     -> original timestamped source rows
     -> joint revision -> timeline tables and character-position tables
```

Times in temporal tables are hours relative to admission (or case presentation when admission is unavailable). UIDs identify event occurrences within a case; they are not patient identifiers and are not globally unique. Preserve the Step 2 outputs when comparing downstream runs: regenerating them creates new random UIDs.

| Script | Purpose and main inputs/outputs |
| --- | --- |
| `prompt_io.py` | Sends `.txt` or `.txt.gz` notes through a tagging prompt; writes tagged model output. Supports a local HTTP model server or direct Transformers generation. |
| `extract_tagged_table.py` | Extracts numbered tags and assigns UIDs; writes headerless `tag_number\|uid4\|mention` tables as `*_tags.bsv.gz`. |
| `temporal_prompt_io.py` | Combines tagged notes and mention tables with a temporal prompt; writes text-only temporal model output. Supports batching mentions for long cases. |
| `extract_temporal_table.py` | Extracts the answer BSV table from temporal model output; writes `*_extracted.bsv.gz`. |
| `mask_cr_append_mt.py` | Reads gzipped tagged notes and tag tables; writes a plain-text UID-tagged note with an appended mention table and a separate `*_uid4.bsv`. |
| `anchor_query_generator.py` | Generates structured-evidence queries for each UID through a local OpenAI-compatible chat endpoint; writes one JSON file per case. |
| `rag_pipeline_llama_embed_folder_summarized_hf.py` | Summarizes each structured event series and embeds it with a local Transformers model; persists Chroma databases and `summary_mapping.json` files linking summaries to original row indices. |
| `anchor_match_ranker_hf.py` | Embeds queries, retrieves candidate summaries, and optionally reranks them with a local causal language model; writes `*_results.json`. |
| `rag_pipeline_llama_embed_folder_summarized.py` | HTTP-server embedding variant; supports `ingest` and an additional `query` action. |
| `anchor_match_ranker.py` | HTTP embedding/reranker variant of candidate matching. |
| `view_match_scores.py` | Flattens match JSON into `*_summary.csv`, with selectable score sorting. |
| `join_anchor_with_source_timed.py` | Joins matched summaries to original structured rows through the summary mapping; uses note/admission metadata to filter rows after discharge plus a 12-hour buffer. Writes `*_joined.csv`. |
| `assign_joint_timestamps.py` | Combines a UID-tagged note, initial temporal table, and joined evidence through a model server; writes revised timeline JSON and failed-response logs. |
| `extract_timelines.py` | Converts timeline JSON into compressed BSV files, one per timeline. |
| `get_tag_charpos_table.py` | Extracts UID-tag spans and optionally joins mentions and temporal estimates; writes position tables. Verify offsets against your text representation before downstream annotation use. |
| `export_timeline_event_time.py` | Exports final timelines 1–3 to two-column BSV (`mention\|time`) and CSV (`event,time`); validates headers and numeric times. |
| `call_lcpp.py` | Shared HTTP helper for llama.cpp's completion endpoint; imported by prompting scripts. |
| `llm_server_utils.sh` | Shared shell functions for starting, waiting for, and stopping llama.cpp servers. |

`templates/` contains tagging (`tts_tag*.txt`), temporal (`temporal_prompt.txt`), query-generation (`aqg_system.txt`, `aqg_uid.txt`), and revision (`ajt.txt`) prompts, plus alternative/historical prompts. For the workflow below, pass the indicated template explicitly: some Python defaults refer to paths that are not included in this snapshot.

## Installation and model services

Use a dedicated Python environment. Python 3.10 or newer is required by type annotations in some scripts; Python 3.11 is an example setup target. The snapshot does not include pinned requirements or an environment lockfile, so the following is a dependency list inferred from imports, not a validated reproduction environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy pandas requests httpx openai \
  torch transformers accelerate \
  langchain-core langchain-openai langchain-chroma langgraph \
  pydantic scikit-learn threadpoolctl
```

Install a PyTorch build compatible with your CUDA environment if using GPUs. GAVEL's workbook tools additionally require `openpyxl`. Model weights, a llama.cpp server executable, MIMIC data, prepared structured tables, and linkage metadata are not included. Prepare the per-subject tables from your authorized data separately; this repository is not a raw-MIMIC ETL package.

The commands below use a locally hosted generation model at `http://127.0.0.1:8080`, with an alias of `uid-model`, and local Qwen3-compatible embedding/reranker snapshots. The tagging/temporal clients send `structured_outputs` regex options; use a compatible server build or adapt those request options for your server. Start your generation server before the model calls, for example:

```bash
/path/to/llama-server \
  -m /path/to/generation-model.gguf -a uid-model \
  --host 127.0.0.1 --port 8080 --jinja --ctx-size 131072
```

Adjust context size, GPU offload, parallel slots, and generation budgets to the model and hardware. The code contains large token limits; long cases may require substantial GPU memory or smaller batches. Keep endpoints inside your approved computing environment. Follow [PhysioNet guidance for responsible use of credentialed data with LLMs](https://physionet.org/news/post/llm-responsible-use/) and your dataset agreement. An OpenAI-compatible endpoint describes the API format; it does not require sending data to OpenAI.

The local HF embedding implementations use normalized last-token pooling. The reranker scores `yes`/`no` with a causal language model. Use architectures appropriate for those implementations, such as the Qwen3 embedding/reranker models used in the scripts; arbitrary sentence encoders or classifier rerankers are not interchangeable. Use the same embedding model, revision, and truncation settings for ingestion and matching.

## Inputs and naming

For an initial run, use one note/encounter per subject, with matching basenames:

```text
/private/project-data/
  notes/
    demo_case.txt
  tabular/
    demo_case.csv.gz
  note_link.csv
```

`demo_case` is an illustrative synthetic identifier, not a supplied patient record. The pipeline can recognize some decorated note filenames, but subject-only fallbacks may select the first matching encounter. If a subject has multiple admissions, validate note-to-admission mapping and candidate selection rather than relying on a bare subject filename.

- **Notes:** UTF-8 plain text or gzip-compressed text, consistently selected with `--input_type`.
- **Structured tables:** one `.csv.gz` per subject, with `t,event,value` columns. A `pt` column is described by the loader but is not required by the summarization code. Use Unix seconds or ISO timestamps with explicit timezone offsets; rows with unparseable timestamps may be skipped. Keep row ordering fixed after ingestion because `summary_mapping.json` refers to row indices.
- **Linkage metadata:** CSV or CSV.GZ with `note_id,subject_id,hadm_id,admittime,dischtime`. Use parseable admission/discharge timestamps and identifiers that match the chosen filenames.
- **BSV:** pipe-separated text. The initial and final temporal schema is `uid4|mention|time|bounds|known|context_uid4s` (some intermediate headers use `context uid4s`). Do not put unescaped `|` in synthetic event labels.

## Run the Python pipeline

Run these commands from the repository root in one Bash session. Replace the illustrative absolute paths with your authorized locations. All generated clinical material, including logs and caches, should stay outside the checkout.

```bash
umask 077
export DATA_ROOT=/private/project-data
export RUN_ROOT=/private/project-runs/run01
export EMBEDDING_MODEL=/path/to/Qwen3-Embedding-snapshot
export RERANKER_MODEL=/path/to/Qwen3-Reranker-snapshot
export GEN_MODEL=uid-model
export MODEL_URL=http://127.0.0.1:8080
export HF_HOME=/private/project-model-cache
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"
export TZ=UTC
mkdir -p "$RUN_ROOT/logs"
```

### 1. Tag event occurrences and assign UIDs

The gzipped tagging output here is intentional: the bundle-preparation script reads `.txt.gz` only.

```bash
python prompt_io.py \
  --in_dir "$DATA_ROOT/notes" --input_type txt \
  --out_dir "$RUN_ROOT/tagged" --out_extension .txt.gz \
  --prompt_file templates/tts_tag.txt \
  --llm_mechanism llama.cpp --endpoint_type openai \
  --lcpp_url "$MODEL_URL" --model_name "$GEN_MODEL" \
  --chunk_overlength --max_input_lines 500 --chunk_line_count 450 \
  > "$RUN_ROOT/logs/tagging.log" 2>&1

python extract_tagged_table.py \
  --input-dir "$RUN_ROOT/tagged" --output-dir "$RUN_ROOT/tags"
```

For this walkthrough, stage plain `.txt` notes in the private input directory. Although the tagger accepts `--input_type txt.gz`, it strips only the final extension when naming outputs, so compressed input produces an extra `.txt` segment and requires corresponding downstream filename changes. Decompress to private staging first to preserve the names used here. Inspect the tagged output and check that each expected case has a nonempty tag table before continuing.

### 2. Estimate and extract the text-only timeline

```bash
python temporal_prompt_io.py \
  --case-dir "$RUN_ROOT/tagged" --tags-dir "$RUN_ROOT/tags" \
  --output-dir "$RUN_ROOT/temporal" \
  --prompt-file templates/temporal_prompt.txt --keep_columns 1 2 \
  --tag_chunk_size 60 --llm_mechanism llama.cpp \
  --endpoint_type openai --lcpp_url "$MODEL_URL" --model_name "$GEN_MODEL" \
  > "$RUN_ROOT/logs/temporal.log" 2>&1

python extract_temporal_table.py \
  --input-dir "$RUN_ROOT/temporal" \
  --output-dir "$RUN_ROOT/temporal_answers" \
  --keep-columns uid4 mention time bounds known "context uid4s" \
  --include-header
```

The extractor handles the model's answer block. The supplied temporal prompt includes a header, so the command keeps it with `--include-header`. Add `--no-header` only for a genuinely headerless model response; otherwise it duplicates the header as a data row. Verify that each output has one header and valid UID rows. Keep the extraction output separate from the temporal input directory because batch extraction attempts to read every directory entry.

### 3. Build the UID bundle and anchor queries

```bash
python mask_cr_append_mt.py \
  --mention_table_dir "$RUN_ROOT/tags" \
  --tagged_case_report_dir "$RUN_ROOT/tagged" \
  --output_dir "$RUN_ROOT/bundle"

python anchor_query_generator.py generate \
  --folder "$RUN_ROOT/bundle" --infix uidtts \
  --host 127.0.0.1 --port 8080 --model-name "$GEN_MODEL" \
  --system-prompt templates/aqg_system.txt \
  --user-prompt templates/aqg_uid.txt \
  > "$RUN_ROOT/logs/anchors.log" 2>&1
```

The generation model may be stopped temporarily during the next HF stages if its GPU memory is needed. Restart it before final revision.

### 4. Embed structured evidence and rank candidates locally

```bash
python rag_pipeline_llama_embed_folder_summarized_hf.py ingest \
  --folder "$DATA_ROOT/tabular" --output-folder "$RUN_ROOT/bundle/tabular" \
  --infix uid_infix --embedding-model "$EMBEDDING_MODEL" \
  --device auto --embedding-batch-size 8 --max-length 8192 \
  > "$RUN_ROOT/logs/ingest.log" 2>&1

python anchor_match_ranker_hf.py compare \
  --folder "$RUN_ROOT/bundle" --anchor-subdir uidtts \
  --rag-subdir tabular/uid_infix \
  --embedding-model "$EMBEDDING_MODEL" --reranker-model "$RERANKER_MODEL" \
  --device auto --embedding-batch-size 8 --reranker-batch-size 4 \
  --max-length 8192 --k 10 \
  > "$RUN_ROOT/logs/ranking.log" 2>&1
```

Reduce batch sizes if memory is insufficient. The HF scripts load models locally and do not call embedding/reranking HTTP services. They enable `trust_remote_code` by default; `--no-trust-remote-code` is available for models supported without it. The ranker accepts legacy HTTP flags for compatibility but ignores them.

For the HTTP alternative, use `rag_pipeline_llama_embed_folder_summarized.py ingest` and `anchor_match_ranker.py compare` with `--host`, `--port`, and `--endpoint-type`. Matching also accepts `--reranker-host` and `--reranker-port`. This route requires separately running compatible embedding and reranker services; `--endpoint-type openai` uses `/v1/embeddings`, while `completion` selects the llama.cpp embedding interface. Do not mix embedding models between ingestion and query matching.

### 5. Join candidates to source rows and revise a case

The following processes the illustrative case named `demo_case`. For a corpus, repeat for each matching result file using the correct subject table and admission mapping.

**Required source configuration:** `assign_joint_timestamps.py` currently hard-codes a model path in both `ChatOpenAI` constructors. Replace both `model=` values with the alias served by your local model (for example, `uid-model`) before this stage. There is no model-name CLI option for this script. Its active branch always uses OpenAI-compatible chat, regardless of `--endpoint-type`; the example selects that interface explicitly.

```bash
CASE_ID=demo_case
COMPARISON_DIR="$RUN_ROOT/bundle/uidtts/comparison_results"

python view_match_scores.py \
  --results-file "$COMPARISON_DIR/${CASE_ID}_results.json" \
  --output "$COMPARISON_DIR/${CASE_ID}_summary.csv" --sort-by score \
  > "$RUN_ROOT/logs/${CASE_ID}_scores.log" 2>&1

python join_anchor_with_source_timed.py \
  --anchor-results "$COMPARISON_DIR/${CASE_ID}_summary.csv" \
  --summary-mapping "$RUN_ROOT/bundle/tabular/uid_infix/uid_infix_${CASE_ID}.csv/summary_mapping.json" \
  --line-level-data "$DATA_ROOT/tabular/${CASE_ID}.csv.gz" \
  --note-adms "$DATA_ROOT/note_link.csv" \
  --output "$COMPARISON_DIR/${CASE_ID}_joined.csv" \
  > "$RUN_ROOT/logs/${CASE_ID}_join.log" 2>&1

python assign_joint_timestamps.py \
  --anchor-results "$COMPARISON_DIR/${CASE_ID}_joined.csv" \
  --case-report "$RUN_ROOT/bundle/${CASE_ID}.txt" \
  --mention-table "$RUN_ROOT/temporal_answers/${CASE_ID}_extracted.bsv.gz" \
  --note-link "$DATA_ROOT/note_link.csv" --template templates/ajt.txt \
  --output "$COMPARISON_DIR/${CASE_ID}_timelines.json" \
  --log-dir "$RUN_ROOT/logs/revision" \
  --host 127.0.0.1 --port 8080 --endpoint-type openai \
  --tau 0.1 --top-k 10 --num-timelines 1 \
  > "$RUN_ROOT/logs/${CASE_ID}_revision.log" 2>&1
```

`tau` filters candidate scores; it is not a clinical confidence threshold. Check admission matching and missing/discarded evidence in the logs. Missing metadata can prevent reliable time normalization or discharge filtering.

### 6. Extract tables and positions

```bash
python extract_timelines.py \
  --directory "$COMPARISON_DIR" --pattern '*_timelines.json' \
  --output-dir "$RUN_ROOT/timelines" --verbose

python get_tag_charpos_table.py \
  --input-dir "$RUN_ROOT/bundle" --pattern '*.txt' \
  --output-dir "$RUN_ROOT/bundle/charpos" --join-mention \
  --timeline-dir "$RUN_ROOT/timelines"

python get_tag_charpos_table.py \
  --input-dir "$RUN_ROOT/bundle" --pattern '*.txt' \
  --output-dir "$RUN_ROOT/bundle/charpos_unrevised" --join-mention \
  --timeline-dir "$RUN_ROOT/temporal_answers"

python export_timeline_event_time.py \
  --timelines-dir "$RUN_ROOT/timelines" --strict
```

The optional final export requires the exact six-column final header and numeric times; unresolved `N/A` times must be handled before this export. It creates `timeline_<n>/` directories with two-column files. Basic extraction and export helpers were checked with invented synthetic data; model inference, GPU execution and Slurm submission have not been validated in this review.


Additional entry-point limitations in this snapshot:

- `prompt_io.py`'s completion branch passes an unsupported `extra_args` argument to `call_lcpp.query_llama_cpp`. Use the OpenAI-compatible route shown above; the completion route needs a source fix.
- `run_pipeline.sh` sources missing `lib/pipeline_utils.sh`, `lib/server_utils.sh`, and `settings.sh`, and invokes a site-specific CUDA setup script. It cannot run as shipped.
- `run_scripts/run_all_uid_pipeline.sh` and `run_scripts/step_1234_tag_time.sh` have Steps 1 and 2 commented out. It expects pre-existing tagged reports and tag tables despite its name. Use the staged workflow above for a fresh run.

[UID_ANCHOR_DETECTION_RUNBOOK.md](UID_ANCHOR_DETECTION_RUNBOOK.md) records additional operational details, but its historical paths, examples, and description of the all-steps launcher should be reconciled with the current source.

## Outputs, reruns, and reproducibility

A completed run normally contains:

```text
run01/
  tagged/                         tagged note/model output
  tags/                           UID mention tables
  temporal/                      raw temporal model output
  temporal_answers/              initial temporal tables
  bundle/
    <case>.txt                    UID-tagged note + mention table
    <case>_uid4.bsv                UID/mention pairs
    tabular/uid_infix/             Chroma databases + summary mappings
    uidtts/
      <case>.json                 anchor queries
      comparison_results/         results, summaries, joins, revised JSON
    charpos/                      revised positions/time tables
    charpos_unrevised/             initial positions/time tables
  timelines/                      final BSVs and optional two-column exports
  logs/                           potentially sensitive run/response logs
```

Most stages skip existing files, but validation differs across scripts; an existing database directory may cause ingestion to skip an incomplete database. For changed data, models, prompts, or settings, use a new private run directory. Preserve existing UIDs when performing a controlled downstream comparison. Inspect failed-case logs, verify UID coverage, and check that every expected case has valid outputs; a process exiting successfully can still have skipped cases.

Record the source revision, Python/package versions, model revisions and quantization, server version/options, prompt versions, data preparation, selection lists, and random seeds. This archive alone does not provide the private study cohort, annotations, exact environment, or all evaluation inputs needed to reproduce every reported result.

## GAVEL pairwise evaluation

`gavel_judge.zip` contains a separate evaluation package. Extract it in a private workspace and read its README:

```bash
unzip gavel_judge.zip -d "$RUN_ROOT/evaluation_tools"
export GAVEL_DIR="$RUN_ROOT/evaluation_tools/gavel_judge"
python -m pip install openpyxl
```

GAVEL needs source notes, prepared structured summaries, and at least two timeline sources. The prompt builder uses corpus folder names and source aliases in `FOLDERS` and `SOURCES` near the top of `make_round_robin_n40.py`; configure those for your corpus. Its default `--pairings remaining` is study-specific, so inspect supported pairings and choose explicitly:

```bash
python "$GAVEL_DIR/make_round_robin_n40.py" \
  --root "$DATA_ROOT/gavel_corpus" \
  --template "$GAVEL_DIR/structured_judge_prompt_2col.txt" --list-pairings
```

Then pass the desired supported pairing name with `--pairings` and an absolute private prompt output directory with `--out`. The builder produces pairwise prompts, a manifest, and a build report. For one generated prompt, use a local judge service and preserve the pairing in the output path:

```bash
python "$GAVEL_DIR/run_judge_gavel.py" \
  --summary_file /private/gavel-prompts/pairing/prompt_demo_case.txt \
  --llm_endpoint http://127.0.0.1:8001 --model_basename judge \
  --log_folder "$RUN_ROOT/judge_outputs" \
  --output_file "$RUN_ROOT/judge_outputs/pairing/demo_case.json"

python "$GAVEL_DIR/make_elo_report.py" \
  --root "$RUN_ROOT/judge_outputs" --out "$RUN_ROOT/gavel_report.html"

python "$GAVEL_DIR/make_review_sample.py" \
  --root "$RUN_ROOT/judge_outputs" --prompts /private/gavel-prompts \
  --out "$RUN_ROOT/manual_review" --reviewers reviewer_a,reviewer_b
```

Replace `pairing` with the actual `<source_a>_vs_<source_b>` directory. Use `--help` to adjust sampling to your corpus. `make_elo_report.py` produces a weighted Bradley–Terry leaderboard with resampling intervals; `make_review_sample.py` writes reviewer workbooks and supporting case material. `manual_review_report.py` summarizes completed review packets and requires explicit reviewer assignments via `--by`.

The bundled `run_judge_gavel.sh` automates judging on Slurm, but requires site-specific configuration. It resolves companion files using `BASH_SOURCE`, which may refer to Slurm's spool directory; adapt path resolution to your submission environment before using it. Judge prompts, raw responses, manifests, reports, and review packets can contain patient-level material and must remain private. The HTML leaderboard embeds case IDs in its JavaScript payload even when the visible charts appear aggregate. GAVEL findings require interpretation and are not a substitute for clinician review.

## Citation

If you use this work, please cite:

Sayantan Kumar, Nicolas Grimaldi, Jack Cummins, and Jeremy C. Weiss. **Anchoring Clinical Events in Time: UID-Preserving Multimodal Reconstruction and Source-Grounded Adjudication.** 2026. [arXiv:2609.13062](https://arxiv.org/abs/2609.13062). Machine Learning for Health Symposium 2026.

```bibtex
@misc{kumar2026anchoring,
  title         = {Anchoring Clinical Events in Time: {UID}-Preserving Multimodal Reconstruction and Source-Grounded Adjudication},
  author        = {Kumar, Sayantan and Grimaldi, Nicolas and Cummins, Jack and Weiss, Jeremy C.},
  year          = {2026},
  eprint        = {2609.13062},
  archivePrefix = {Machine Learning for Health Symposium (ML4H 2026)},
  doi           = {10.48550/arXiv.2609.13062},
  url           = {https://arxiv.org/abs/2609.13062}
}
```
