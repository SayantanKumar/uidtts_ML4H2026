#!/bin/bash
# ============================================================================
# Tag-Then-Time Multimodal Pipeline
# ============================================================================
# Main pipeline script. Configuration and utility functions are loaded from
# separate files for easier maintenance.
# ============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Source libraries
source "$SCRIPT_DIR/lib/pipeline_utils.sh"
source "$SCRIPT_DIR/lib/server_utils.sh"

# Source default settings
source "$SCRIPT_DIR/settings.sh"

# Override with local settings if they exist
if [[ -f "$SCRIPT_DIR/settings.local.sh" ]]; then
    log_info "Loading local settings from settings.local.sh"
    source "$SCRIPT_DIR/settings.local.sh"
fi

# Source CUDA environment
source /usr/bin/fac-switch.sh cuda 12.8.1

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
SKIP_SERVERS=false
STOP_AT_END=false
KEEP_EMBEDDINGS=false
DRY_RUN=false
PARALLEL_RAG=false
PORT=""
INSTRUCT_MODEL=""
REASONING_MODEL=""
CUDA_DEVICES="${CUDA_DEVICES:-}"
START_STEP=1
END_STEP=6

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-servers)
            SKIP_SERVERS=true
            shift
            ;;
        --stop-at-end)
            STOP_AT_END=true
            shift
            ;;
        --keep-embeddings)
            KEEP_EMBEDDINGS=true
            shift
            ;;
        --parallel-rag)
            PARALLEL_RAG=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --embedding-port)
            EMBEDDING_PORT="$2"
            shift 2
            ;;
        --instruct-model)
            INSTRUCT_MODEL="$2"
            shift 2
            ;;
        --reasoning-model)
            REASONING_MODEL="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --config)
            source "$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --debug)
            export DEBUG=true
            shift
            ;;
        --start-step)
            START_STEP="$2"
            shift 2
            ;;
        --end-step)
            END_STEP="$2"
            shift 2
            ;;
        --steps)
            IFS='-' read -ra STEPS_RANGE <<< "$2"
            START_STEP="${STEPS_RANGE[0]}"
            END_STEP="${STEPS_RANGE[1]:-${STEPS_RANGE[0]}}"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validate Arguments
# -----------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    log_error "Missing required argument: input_folder"
    show_help
    exit 1
fi

# Set final values with defaults
INPUT_FOLDER="$1"
OUTPUT_FOLDER="${2:-output}"
PORT="${PORT:-$DEFAULT_PORT}"
EMBEDDING_PORT="${EMBEDDING_PORT:-$((${PORT} + 1))}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-$DEFAULT_INSTRUCT_MODEL}"
REASONING_MODEL="${REASONING_MODEL:-$DEFAULT_REASONING_MODEL}"
CUDA_DEVICES="${CUDA_DEVICES:-$CUDA_DEVICES}"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

ENDPOINT_TYPE=$(get_endpoint_type)

# Record start time
START_TIME=$(date +%s)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

# Calculate mem-fraction-static from absolute GB amount and tensor parallelism
calculate_mem_fraction() {
    local mem_gb=$1
    local tp_size=$2

    # Get available (free) GPU memory in GB from first visible GPU
    local gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
    local free_mem_mb=$(nvidia-smi --id=$gpu_id --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo "140000")
    local free_mem_gb=$((free_mem_mb / 1024))

    # Calculate mem-fraction-static: (mem_gb / tp_size) / free_mem_gb
    # This ensures we allocate based on available memory, not total capacity
    # Using bc for floating point arithmetic
    local mem_per_gpu=$(echo "scale=6; $mem_gb / $tp_size" | bc)
    local mem_fraction=$(echo "scale=6; $mem_per_gpu / $free_mem_gb" | bc)

    echo $mem_fraction
}

# -----------------------------------------------------------------------------
# Display Configuration
# -----------------------------------------------------------------------------
log_info "============================================"
log_info "Pipeline Configuration"
log_info "============================================"
log_info "Input folder:    $INPUT_FOLDER"
log_info "Output folder:   $OUTPUT_FOLDER"
log_info "Port:            $PORT"
log_info "Embedding port:  $EMBEDDING_PORT"
log_info "Instruct model:  $(basename "$INSTRUCT_MODEL")"
log_info "Reasoning model: $(basename "$REASONING_MODEL")"
log_info "CUDA devices:    $CUDA_DEVICES"
log_info "API mode:        $LLAMA_API_MODE"
log_info "Endpoint type:   $ENDPOINT_TYPE"
log_info "Concurrency:     $CONCURRENCY"
log_info "Skip servers:    $SKIP_SERVERS"
log_info "Stop at end:     $STOP_AT_END"
log_info "Keep embeddings: $KEEP_EMBEDDINGS"
log_info "Parallel RAG:    $PARALLEL_RAG"
log_info "Dry run:         $DRY_RUN"
log_info "============================================"

# Dry run - just show config and exit
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Dry run complete. No actions taken."
    exit 0
fi

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
check_prerequisites || exit 1

if ! check_required_dir "$INPUT_FOLDER" "input folder"; then
    exit 1
fi

# Check model files exist
if [[ "$SKIP_SERVERS" == "false" ]]; then
    check_required_file "$INSTRUCT_MODEL" "instruct model" || exit 1
    check_required_file "$REASONING_MODEL" "reasoning model" || exit 1
fi

# Setup output directories
setup_output_dirs "$OUTPUT_FOLDER"

# Change to parent directory for relative script paths
cd "$PARENT_DIR"

TOTAL_STEPS=6

# =============================================================================
# PARALLEL RAG: Start embedding server and RAG ingestion early
# =============================================================================
RAG_PID=""
EMBEDDING_PID=""

# Only start parallel RAG if we're running step 5 (which needs embeddings)
if [[ "$PARALLEL_RAG" == "true" && $START_STEP -le 5 && $END_STEP -ge 5 ]]; then
    log_info "Starting RAG ingestion in parallel with main pipeline..."

    # Check if embeddings already exist for all patients
    RAG_OUTPUT_DIR="$MTCR_FOLDER/tabular/uid_infix"
    patient_count=$(ls "$INPUT_FOLDER/tabular/"*.csv.gz 2>/dev/null | wc -l)
    embedding_count=$(ls -d "$RAG_OUTPUT_DIR"/uid_infix_*.csv 2>/dev/null | wc -l)

    if [[ -d "$RAG_OUTPUT_DIR" && $embedding_count -ge $patient_count ]]; then
        log_info "Embeddings already exist for all patients ($embedding_count/$patient_count), skipping parallel RAG"
    else
        if [[ $embedding_count -gt 0 ]]; then
            log_warning "Embeddings incomplete ($embedding_count/$patient_count patients), running RAG ingestion for missing patients"
        fi
        # Start embedding server
        log_info "Starting embedding server on port $EMBEDDING_PORT..."

        # Check if we should use sglang or llama.cpp for embeddings
        if [[ "${EMBEDDING_SERVER_TYPE:-llama}" == "sglang" ]]; then
            log_info "Using sglang for embedding server"

            # Calculate mem-fraction-static from absolute GB amount
            EMBEDDING_MEM_FRACTION=$(calculate_mem_fraction "${EMBEDDING_MEM_GB:-20}" "${EMBEDDING_TP:-8}")
            log_info "Embedding server: ${EMBEDDING_MEM_GB:-20}GB across ${EMBEDDING_TP:-8} GPUs = mem-fraction-static=$EMBEDDING_MEM_FRACTION"

            # Use Singularity if available, otherwise use SGLANG_PYTHON
            if [[ -n "${SINGULARITY_IMAGE:-}" && -f "${SINGULARITY_BIN:-}" ]]; then
                log_info "Starting embedding server in Singularity container"
                CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" TORCHINDUCTOR_CACHE_DIR=/data/weissjc/.cache/vllm/vllm_cache \
                    "$SINGULARITY_BIN" exec --nv -B /data/weissjc/.cache/ "$SINGULARITY_IMAGE" \
                    sglang serve \
                    --model-path "$DEFAULT_EMBEDDING_MODEL" \
                    --tp "${EMBEDDING_TP:-8}" \
                    --is-embedding \
                    --host 0.0.0.0 \
                    --port "$EMBEDDING_PORT" \
                    --trust-remote-code \
                    --disable-radix-cache \
                    --mem-fraction-static "$EMBEDDING_MEM_FRACTION" \
                    >> "$LOG_DIR/embedding_server.log" 2>&1 &
            else
                CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" $SGLANG_PYTHON -m sglang.launch_server \
                    --model-path "$DEFAULT_EMBEDDING_MODEL" \
                    --is-embedding \
                    --host 0.0.0.0 \
                    --port "$EMBEDDING_PORT" \
                    --trust-remote-code \
                    --disable-radix-cache \
                    >> "$LOG_DIR/embedding_server.log" 2>&1 &
            fi
        else
            log_info "Using llama.cpp for embedding server"
            CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" $LLAMA_EMBEDDING_BIN \
                -m "$DEFAULT_EMBEDDING_MODEL" \
                -t 4 -ngl 999 -n 480000 --ctx-size 60000 -np 8 \
                --temp 0.6 --cache-reuse 256 \
                --port "$EMBEDDING_PORT" --jinja \
                --pooling last --embedding --no-context-shift \
                >> "$LOG_DIR/embedding_server.log" 2>&1 &
        fi
        EMBEDDING_PID=$!

        # Wait for embedding server
        EMBEDDING_TIMEOUT=${EMBEDDING_TIMEOUT:-150}
        START_WAIT=$(date +%s)
        until curl -s "http://localhost:$EMBEDDING_PORT" > /dev/null 2>&1; do
            NOW=$(date +%s)
            if (( NOW - START_WAIT > EMBEDDING_TIMEOUT )); then
                log_error "Embedding server startup timeout"
                kill $EMBEDDING_PID 2>/dev/null || true
                exit 1
            fi
            sleep 1
        done
        log_success "Embedding server ready (PID: $EMBEDDING_PID)"

        # Run RAG pipeline in background
        log_info "Starting RAG ingestion in background..."
        python $SCRIPT_RAG_PIPELINE \
            --folder "$INPUT_FOLDER/tabular/" \
            --output-folder "$MTCR_FOLDER/tabular/" \
            --infix "uid_infix" \
            ingest \
            --port "$EMBEDDING_PORT" \
            --endpoint-type "$ENDPOINT_TYPE" \
            >> "$LOG_DIR/rag_ingestion.log" 2>&1 &
        RAG_PID=$!
        log_info "RAG ingestion started (PID: $RAG_PID), logs: $LOG_DIR/rag_ingestion.log"
    fi
fi

# =============================================================================
# STEP 1: Generate tagged mentions
# =============================================================================
if [[ $START_STEP -le 1 && $END_STEP -ge 1 ]]; then
    log_step 1 $TOTAL_STEPS "Generating tagged mentions"

    if [[ "$SKIP_SERVERS" == "false" ]]; then
        CURRENT_SERVER_PID=$(start_server "$INSTRUCT_MODEL" "$PORT" "instruct")
        if [[ $? -ne 0 ]]; then
            log_error "Failed to start instruct model server"
            exit 1
        fi
    fi

    NOTHINK=$(get_nothink_param "$INSTRUCT_MODEL")

    pids=()
    for i in $(seq 1 "$CONCURRENCY"); do
        python $SCRIPT_PROMPT_IO \
            --in_dir "$INPUT_FOLDER" \
            --input_type "txt.gz" \
            --out_dir "$OUTPUT_FOLDER/tagged" \
            --prompt_file "$TEMPLATE_TAG" \
            --llm_mechanism llama.cpp \
            --lcpp_url "http://localhost:$PORT" \
            --model_name "$INSTRUCT_MODEL" \
            --endpoint_type "$ENDPOINT_TYPE" \
            --extra_body "$NOTHINK" \
            --out_extension ".txt.gz" &
        pids+=("$!")
    done

    set +e
    for pid in "${pids[@]}"; do
        wait "$pid" || log_warning "Process $pid had non-zero exit"
    done
    set -e

    if [[ "$SKIP_SERVERS" == "false" ]]; then
        cleanup_server "$CURRENT_SERVER_PID" "$PORT"
    fi
fi

# =============================================================================
# STEP 2: Extract tags
# =============================================================================
if [[ $START_STEP -le 2 && $END_STEP -ge 2 ]]; then
    log_step 2 $TOTAL_STEPS "Extracting tags"

    python $SCRIPT_EXTRACT_TAGS \
        --input-dir "$OUTPUT_FOLDER/tagged" \
        --output-dir "$OUTPUT_FOLDER/tags"
fi

# =============================================================================
# STEP 3: Generate temporal tables
# =============================================================================
if [[ $START_STEP -le 3 && $END_STEP -ge 3 ]]; then
    log_step 3 $TOTAL_STEPS "Generating temporal tables"

    if [[ "$SKIP_SERVERS" == "false" ]]; then
        CURRENT_SERVER_PID=$(start_server "$REASONING_MODEL" "$PORT" "reasoning")
        if [[ $? -ne 0 ]]; then
            log_error "Failed to start reasoning model server"
            exit 1
        fi
    fi

    pids=()
    for i in $(seq 1 "$CONCURRENCY"); do
        python $SCRIPT_TEMPORAL_IO \
            --case-dir "$OUTPUT_FOLDER/tagged" \
            --tags-dir "$OUTPUT_FOLDER/tags" \
            --output-dir "$OUTPUT_FOLDER/temporal" \
            --prompt-file "$TEMPLATE_TEMPORAL" \
            --llm_mechanism llama.cpp \
            --lcpp_url "http://localhost:$PORT" \
            --keep_columns 1 2 \
            --endpoint_type openai \
            --model_name "$REASONING_MODEL" &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || log_warning "Process $pid had non-zero exit"
    done

    if [[ "$SKIP_SERVERS" == "false" ]]; then
        cleanup_server "$CURRENT_SERVER_PID" "$PORT"
    fi
fi

# =============================================================================
# STEP 4: Extract mention and time tables
# =============================================================================
if [[ $START_STEP -le 4 && $END_STEP -ge 4 ]]; then
    log_step 4 $TOTAL_STEPS "Extracting mention and time tables"

    python $SCRIPT_EXTRACT_TEMPORAL \
        --input-dir "$OUTPUT_FOLDER/temporal" \
        --output-dir "$OUTPUT_FOLDER/temporal/answers" \
        --keep-columns uid4 mention time bounds known "context uid4s" \
        --include-header \
        --no-header
fi

# =============================================================================
# STEP 5: Anchor query generation and RAG
# =============================================================================
if [[ $START_STEP -le 5 && $END_STEP -ge 5 ]]; then
    log_step 5 $TOTAL_STEPS "Anchor query generation and RAG"

    if [[ "$SKIP_SERVERS" == "false" ]]; then
        CURRENT_SERVER_PID=$(start_server "$INSTRUCT_MODEL" "$PORT" "instruct")
        if [[ $? -ne 0 ]]; then
            log_error "Failed to start instruct model server"
            exit 1
        fi
    fi

    # Prepare case reports
    python $SCRIPT_MASK_CR \
        --mention_table_dir "$OUTPUT_FOLDER/tags/" \
        --tagged_case_report_dir "$OUTPUT_FOLDER/tagged/" \
        --output_dir "$MTCR_FOLDER"

    # Generate anchor queries
    pids=()
    for i in $(seq 1 "$CONCURRENCY"); do
        python $SCRIPT_ANCHOR_QUERY \
            generate \
            --folder "$MTCR_FOLDER" \
            --host localhost \
            --port "$PORT" \
            --infix "$DEFAULT_AQG_INFIX" \
            --system-prompt "$TEMPLATE_AQG_SYSTEM" \
            --user-prompt "$TEMPLATE_AQG_USER" \
            --model-name "$REASONING_MODEL" &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || log_warning "Process $pid had non-zero exit"
    done

    if [[ "$SKIP_SERVERS" == "false" ]]; then
        cleanup_server "$CURRENT_SERVER_PID" "$PORT"
    fi

    # Check if candidates already exist (for resume/skip embedding logic)
    mkdir -p "$COMPARISON_DIR"
    EXPECTED_JOINED=$(ls "$MTCR_FOLDER"/*.txt.txt 2>/dev/null | wc -l)
    EXISTING_JOINED=$(ls "$COMPARISON_DIR"/*_joined.csv 2>/dev/null | wc -l)
    SKIP_EMBEDDINGS=false

    if [[ "$EXISTING_JOINED" -ge "$EXPECTED_JOINED" && "$EXPECTED_JOINED" -gt 0 ]]; then
        log_info "All $EXISTING_JOINED candidate files already exist, skipping embedding step"
        SKIP_EMBEDDINGS=true
    fi

    if [[ "$SKIP_EMBEDDINGS" == "false" ]]; then
        # If RAG was started in parallel, wait for it to complete
        if [[ -n "$RAG_PID" ]]; then
            log_info "Waiting for parallel RAG ingestion to complete (PID: $RAG_PID)..."
            wait "$RAG_PID"
            RAG_EXIT_CODE=$?
            if [[ $RAG_EXIT_CODE -ne 0 ]]; then
                log_error "Parallel RAG ingestion failed with exit code $RAG_EXIT_CODE"
                log_error "Check logs: $LOG_DIR/rag_ingestion.log"
                kill $EMBEDDING_PID 2>/dev/null || true
                exit 1
            fi
            log_success "Parallel RAG ingestion completed"
        else
            # Start embedding server (always started, regardless of --skip-servers)
            log_info "Starting embedding server..."

            # Check if we should use sglang or llama.cpp for embeddings
            if [[ "${EMBEDDING_SERVER_TYPE:-llama}" == "sglang" ]]; then
                log_info "Using sglang for embedding server"
                CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" $SGLANG_PYTHON -m sglang.launch_server \
                    --model-path "$DEFAULT_EMBEDDING_MODEL" \
                    --is-embedding \
                    --host 0.0.0.0 \
                    --port "$EMBEDDING_PORT" \
                    --trust-remote-code \
                    --disable-radix-cache \
                    >> "$LOG_DIR/embedding_server.log" 2>&1 &
            else
                log_info "Using llama.cpp for embedding server"
                CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" $LLAMA_EMBEDDING_BIN \
                    -m "$DEFAULT_EMBEDDING_MODEL" \
                    -t 4 -ngl 999 -n 480000 --ctx-size 60000 -np 8 \
                    --temp 0.6 --cache-reuse 256 \
                    --port "$EMBEDDING_PORT" --jinja \
                    --pooling last --embedding --no-context-shift \
                    >> "$LOG_DIR/embedding_server.log" 2>&1 &
            fi
            EMBEDDING_PID=$!

            # Wait for embedding server
            EMBEDDING_TIMEOUT=${EMBEDDING_TIMEOUT:-150}
            START_WAIT=$(date +%s)
            until curl -s "http://localhost:$EMBEDDING_PORT" > /dev/null 2>&1; do
                NOW=$(date +%s)
                if (( NOW - START_WAIT > EMBEDDING_TIMEOUT )); then
                    log_error "Embedding server startup timeout"
                    kill $EMBEDDING_PID 2>/dev/null || true
                    exit 1
                fi
                sleep 1
            done
            log_success "Embedding server ready (PID: $EMBEDDING_PID)"

            # Start reranker server
            log_info "Starting reranker server..."

            # Calculate mem-fraction-static from absolute GB amount
            RERANKER_MEM_FRACTION=$(calculate_mem_fraction "${RERANKER_MEM_GB:-20}" "${RERANKER_TP:-8}")
            log_info "Reranker server: ${RERANKER_MEM_GB:-20}GB across ${RERANKER_TP:-8} GPUs = mem-fraction-static=$RERANKER_MEM_FRACTION"

            # Use Singularity if available, otherwise use SGLANG_PYTHON
            if [[ -n "${SINGULARITY_IMAGE:-}" && -f "${SINGULARITY_BIN:-}" ]]; then
                log_info "Starting reranker server in Singularity container"
                CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" TORCHINDUCTOR_CACHE_DIR=/data/weissjc/.cache/vllm/vllm_cache \
                    "$SINGULARITY_BIN" exec --nv -B /data/weissjc/.cache/ "$SINGULARITY_IMAGE" \
                    sglang serve \
                    --model-path "$DEFAULT_RERANKER_MODEL" \
                    --tp "${RERANKER_TP:-8}" \
                    --host 0.0.0.0 \
                    --port "$RERANKER_PORT" \
                    --trust-remote-code \
                    --mem-fraction-static "$RERANKER_MEM_FRACTION" \
                    --chat-template /data/weissjc/sgl/sglang/examples/chat_template/qwen3_reranker.jinja \
                    >> "$LOG_DIR/reranker_server.log" 2>&1 &
            else
                CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" $SGLANG_PYTHON -m sglang.launch_server \
                    --model-path "$DEFAULT_RERANKER_MODEL" \
                    --host 0.0.0.0 \
                    --port "$RERANKER_PORT" \
                    --trust-remote-code \
                    --chat-template /data/weissjc/sgl/sglang/examples/chat_template/qwen3_reranker.jinja \
                    >> "$LOG_DIR/reranker_server.log" 2>&1 &
            fi
            RERANKER_PID=$!

            # Wait for reranker server
            RERANKER_TIMEOUT=${RERANKER_TIMEOUT:-150}
            START_WAIT=$(date +%s)
            until curl -s "http://localhost:$RERANKER_PORT" > /dev/null 2>&1; do
                NOW=$(date +%s)
                if (( NOW - START_WAIT > RERANKER_TIMEOUT )); then
                    log_error "Reranker server startup timeout"
                    kill $EMBEDDING_PID 2>/dev/null || true
                    kill $RERANKER_PID 2>/dev/null || true
                    exit 1
                fi
                sleep 1
            done
            log_success "Reranker server ready (PID: $RERANKER_PID)"

            # Run RAG pipeline
            python $SCRIPT_RAG_PIPELINE \
                --folder "$INPUT_FOLDER/tabular/" \
                --output-folder "$MTCR_FOLDER/tabular/" \
                --infix "uid_infix" \
                ingest \
                --port "$EMBEDDING_PORT" \
                --endpoint-type "$ENDPOINT_TYPE"
        fi

        # Run ranker
        python $SCRIPT_ANCHOR_RANKER \
            compare \
            --folder "$MTCR_FOLDER" \
            --anchor-subdir "$DEFAULT_AQG_INFIX" \
            --rag-subdir "tabular/uid_infix/" \
            --host localhost \
            --port "$EMBEDDING_PORT" \
            --reranker-port "$RERANKER_PORT" \
            --k 10 \
            --endpoint-type "$ENDPOINT_TYPE"

        # Kill servers (always started, so always killed)
        log_info "Stopping embedding and reranker servers..."
        kill $EMBEDDING_PID 2>/dev/null || true
        kill $RERANKER_PID 2>/dev/null || true

        # Process results
        ls "$COMPARISON_DIR"/*_results.json 2>/dev/null | shuf | while IFS= read -r results_file; do
            if [ -f "$results_file" ]; then
                base_name=$(basename "$results_file" _results.json)

                # Generate summary
                python $SCRIPT_VIEW_SCORES \
                    --results-file "$results_file" \
                    --output "$COMPARISON_DIR/${base_name}_summary.csv" \
                    --sort-by score

                # Extract patient ID
                patient_id=$(echo "$base_name" | cut -d'-' -f1)

                # Join with source
                python $SCRIPT_JOIN_ANCHOR \
                    --anchor-results "$COMPARISON_DIR/${base_name}_summary.csv" \
                    --summary-mapping "$MTCR_FOLDER/tabular/uid_infix/uid_infix_${patient_id}.csv/summary_mapping.json" \
                    --line-level-data "$INPUT_FOLDER/tabular/${patient_id}.csv.gz" \
                    --output "$COMPARISON_DIR/${base_name}_joined.csv" \
                    --note-adms "$NOTE_LINK_PATH"
            fi
        done

        # Clean up embeddings if not keeping them
        if [[ "$KEEP_EMBEDDINGS" == "false" ]]; then
            log_info "Removing embeddings to save space..."
            rm -rf "$MTCR_FOLDER/tabular/uid_infix/"
            log_success "Embeddings removed"
        fi
    fi
fi

# =============================================================================
# STEP 6: Final timestamp revision
# =============================================================================
if [[ $START_STEP -le 6 && $END_STEP -ge 6 ]]; then
    log_step 6 $TOTAL_STEPS "Final timestamp revision"

    if [[ "$SKIP_SERVERS" == "false" ]]; then
        CURRENT_SERVER_PID=$(start_server "$REASONING_MODEL" "$PORT" "reasoning")
        if [[ $? -ne 0 ]]; then
            log_error "Failed to start reasoning model server"
            exit 1
        fi
    fi

    pids=()
    for joined_file in "$COMPARISON_DIR"/*_joined.csv; do
        if [ -f "$joined_file" ]; then
            base_name=$(basename "$joined_file" _joined.csv)
            patient_id=$(echo "$base_name" | sed 's/\.txt$//' | sed 's/\.txt\.gz$//')

            case_report_file="$MTCR_FOLDER/${patient_id}.txt.txt"
            [ ! -f "$case_report_file" ] && case_report_file="$MTCR_FOLDER/${patient_id}.txt.gz"

            mention_table_file="$OUTPUT_FOLDER/temporal/answers/${patient_id}_extracted.bsv.gz"

            if [ -f "$case_report_file" ] && [ -f "$mention_table_file" ]; then
                python $SCRIPT_ASSIGN_TIMESTAMPS \
                    --anchor-results "$joined_file" \
                    --case-report "$case_report_file" \
                    --mention-table "$mention_table_file" \
                    --output "$COMPARISON_DIR/${base_name}_timelines.json" \
                    --tau "$TAU" \
                    --host localhost \
                    --port "$PORT" \
                    --template "$TEMPLATE_AJT" \
                    --note-link "$NOTE_LINK_PATH" \
                    --endpoint-type "$ENDPOINT_TYPE" &
                pids+=("$!")

                if [ "${#pids[@]}" -eq "$CONCURRENCY" ]; then
                    for pid in "${pids[@]}"; do
                        wait "$pid" || true
                    done
                    pids=()
                fi
            fi
        fi
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done

    if [[ "$SKIP_SERVERS" == "false" ]]; then
        cleanup_server "$CURRENT_SERVER_PID" "$PORT"
    fi

    # =============================================================================
    # Post-processing
    # =============================================================================
    log_info "Running post-processing..."

    python $SCRIPT_EXTRACT_TIMELINES \
        --directory "$COMPARISON_DIR" \
        --pattern "*_timelines.json" \
        -o "$OUTPUT_FOLDER/timelines/" \
        -v

    python $SCRIPT_GET_CHARPOS \
        --input-dir "$OUTPUT_FOLDER/bundle/" \
        --pattern "*.txt.txt" \
        --output-dir "$OUTPUT_FOLDER/bundle/charpos" \
        --join-mention \
        --timeline-dir "$OUTPUT_FOLDER/timelines"

    python $SCRIPT_GET_CHARPOS \
        --input-dir "$OUTPUT_FOLDER/bundle/" \
        --pattern "*.txt.txt" \
        --output-dir "$OUTPUT_FOLDER/bundle/charpos_unrevised" \
        --join-mention \
        --timeline-dir "$OUTPUT_FOLDER/temporal/answers"

    for file in "$OUTPUT_FOLDER"/bundle/charpos_unrevised/*; do
        [ -f "$file" ] && mv "$file" "${file%_extracted.bsv}_1.bsv"
    done
fi

# =============================================================================
# Summary
# =============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

show_summary "$OUTPUT_FOLDER" "${DURATION_MIN}m ${DURATION_SEC}s"

# =============================================================================
# Optional: Stop servers at end
# =============================================================================
if [[ "$STOP_AT_END" == "true" ]]; then
    log_info "Stopping servers on port $PORT (--stop-at-end)..."
    # Kill any process listening on the main server port
    fuser -k "$PORT/tcp" 2>/dev/null || true
    log_success "Servers stopped"
fi
