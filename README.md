
# Textual-Tabular Alignment

This repository contains tools for aligning clinical events described in
free-text notes with timestamped events from structured electronic health record
(EHR) tables.

The central use case is temporal alignment: a clinical note often describes
events such as symptoms, treatments, diagnoses, transfers, procedures, and
outcomes, but those mentions may be relative, incomplete, or ambiguous in time.
Structured EHR data contains timestamped observations and interventions, but it
is often difficult to know which rows correspond to which textual events. This
project connects the two views.

## Motivation

Clinical timelines are usually distributed across two complementary sources:

- **Textual data**, which gives rich clinical context but often describes time
  indirectly.
- **Tabular data**, which provides timestamps and structured values but lacks
  narrative context.

The goal of this project is to use language models and retrieval methods to
bridge those sources. The pipeline extracts clinical event mentions from notes,
assigns each mention a stable identifier, estimates an initial text-only
timeline, retrieves relevant structured evidence, and produces a revised final
timeline grounded in both text and tabular data.

## High-Level Workflow

```text
clinical note
  -> event mention tagging
  -> stable mention identifiers
  -> initial text-only temporal table
  -> anchor query generation
  -> retrieval over patient-specific tabular data
  -> matched timestamped evidence
  -> final revised timeline
  -> timeline and character-position outputs
```

## Pipeline Stages

### 1. Event Mention Tagging

The pipeline first identifies clinically meaningful event mentions in each note.
These may include symptoms, diagnoses, medications, procedures, hospital events,
dispositions, complications, or other timeline-relevant events.

### 2. Stable Mention Identification

Each extracted event mention is assigned a stable short identifier. These
identifiers allow the same event mention to be tracked across all later steps:
initial temporal reasoning, structured-data retrieval, final timeline revision,
and character-position output.

### 3. Initial Text-Only Temporal Reasoning

A language model estimates when each event occurred relative to admission or
case presentation. This creates an initial temporal table from the note alone.
The result is useful but may still be uncertain because narrative time can be
ambiguous or incomplete.

### 4. Structured Evidence Querying

For each textual event mention, the system generates natural-language anchor
queries describing what structured EHR evidence might help determine the event's
time. For example, a medication mention may map to medication administration
rows, while a procedure mention may map to procedure or order records.

### 5. Retrieval Over Tabular EHR Data

Patient-specific tabular records are summarized, embedded, and searched using
the generated anchor queries. Retrieved candidates are reranked so that each
textual event is paired with the most relevant timestamped structured evidence.

### 6. Final Timeline Revision

The final reasoning step combines the original note, the initial text-only
timeline, and the retrieved structured evidence. The output is a revised
timeline that can include multiple plausible timeline interpretations when the
case remains ambiguous.

### 7. Post-Processing

Final timeline JSON outputs are converted into tabular timeline files. The
pipeline also produces character-position tables that link timeline rows back to
the original tagged mention spans in the note.

## Main Module

The current end-to-end implementation lives in:

```text
uid_anchor_detection/
```

That module includes:

- prompt templates for event tagging, anchor-query generation, and timeline
  revision
- Python scripts for extraction, retrieval, reranking, joining, and
  post-processing
- Biowulf/Slurm batch scripts for stepwise or full-pipeline execution
- a detailed operational runbook

For implementation details and execution instructions, see:

```text
uid_anchor_detection/README.md
uid_anchor_detection/UID_ANCHOR_DETECTION_RUNBOOK.md
```

## Expected Inputs

At a high level, the pipeline expects:

- clinical notes as plain text files
- patient-specific structured EHR tables
- a note-link/admission metadata file connecting notes to patient encounters
- locally available language models and embedding/reranking models

The structured EHR tables are expected to contain timestamped events and values
for each patient or subject.

## Expected Outputs

The pipeline produces:

- tagged clinical notes
- stable event mention tables
- initial text-only temporal tables
- anchor queries for structured evidence
- retrieved/reranked structured evidence matches
- joined timestamped evidence rows
- final revised timelines
- character-position tables linking timeline rows to text spans


