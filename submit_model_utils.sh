#!/bin/bash

# Shared model argument handling for the submit wrappers.

DEFAULT_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
DEFAULT_MODEL_ALIAS="qwen35"

show_model_submit_help() {
    local wrapper_name="$1"

    cat <<EOF
Usage:
  bash $wrapper_name [model options]

Model options:
  --model-path PATH              Use PATH for both instruct and reasoning jobs
  --model-alias NAME             Use NAME for both model aliases
  --instruct-model-path PATH     Set only the instruct model path
  --reasoning-model-path PATH    Set only the reasoning model path
  --instruct-model-alias NAME    Set only the instruct model alias
  --reasoning-model-alias NAME   Set only the reasoning model alias
  --notes-dir DIR                Folder containing source note files
  --structured-dir DIR           Folder containing structured .csv.gz files
  --note-link FILE               Note/admission mapping CSV
  --gpu-type TYPE                Slurm GPU type for LLM jobs
  --gpu-count N                  Number of GPUs for LLM jobs
  --output-name NAME             Write beneath PROJECT_DIR/outputs/NAME
  --output-dir DIR               Use an explicit output directory
  --help                         Show this help

When the aliases differ, the default output name is
INSTRUCT_ALIAS-to-REASONING_ALIAS.

Example:
  bash $wrapper_name --model-path /data/models/GLM-5.2-00001-of-00006.gguf --model-alias glm52
EOF
}

require_model_option_value() {
    local option="$1"
    local value="${2:-}"

    if [[ -z "$value" || "$value" == --* ]]; then
        echo "Missing value for $option" >&2
        exit 2
    fi
}

configure_model_submission() {
    local wrapper_name="$1"
    shift

    INSTRUCT_MODEL_PATH="${INSTRUCT_MODEL_PATH:-$DEFAULT_MODEL_PATH}"
    REASONING_MODEL_PATH="${REASONING_MODEL_PATH:-$DEFAULT_MODEL_PATH}"
    INSTRUCT_MODEL_ALIAS="${INSTRUCT_MODEL_ALIAS:-$DEFAULT_MODEL_ALIAS}"
    REASONING_MODEL_ALIAS="${REASONING_MODEL_ALIAS:-$DEFAULT_MODEL_ALIAS}"
    MODEL_OUTPUT_NAME="${MODEL_OUTPUT_NAME:-}"
    MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-}"
    INPUT_DIR="${INPUT_DIR:-}"
    TABULAR_DIR="${TABULAR_DIR:-}"
    NOTE_LINK_PATH="${NOTE_LINK_PATH:-}"
    LLM_GPU_TYPE="${LLM_GPU_TYPE:-a100}"
    LLM_GPU_COUNT="${LLM_GPU_COUNT:-2}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-path)
                require_model_option_value "$1" "${2:-}"
                INSTRUCT_MODEL_PATH="$2"
                REASONING_MODEL_PATH="$2"
                shift 2
                ;;
            --model-alias)
                require_model_option_value "$1" "${2:-}"
                INSTRUCT_MODEL_ALIAS="$2"
                REASONING_MODEL_ALIAS="$2"
                shift 2
                ;;
            --instruct-model-path)
                require_model_option_value "$1" "${2:-}"
                INSTRUCT_MODEL_PATH="$2"
                shift 2
                ;;
            --reasoning-model-path)
                require_model_option_value "$1" "${2:-}"
                REASONING_MODEL_PATH="$2"
                shift 2
                ;;
            --instruct-model-alias)
                require_model_option_value "$1" "${2:-}"
                INSTRUCT_MODEL_ALIAS="$2"
                shift 2
                ;;
            --reasoning-model-alias)
                require_model_option_value "$1" "${2:-}"
                REASONING_MODEL_ALIAS="$2"
                shift 2
                ;;
            --notes-dir)
                require_model_option_value "$1" "${2:-}"
                INPUT_DIR="$2"
                shift 2
                ;;
            --structured-dir|--tabular-dir)
                require_model_option_value "$1" "${2:-}"
                TABULAR_DIR="$2"
                shift 2
                ;;
            --note-link)
                require_model_option_value "$1" "${2:-}"
                NOTE_LINK_PATH="$2"
                shift 2
                ;;
            --gpu-type)
                require_model_option_value "$1" "${2:-}"
                LLM_GPU_TYPE="$2"
                shift 2
                ;;
            --gpu-count)
                require_model_option_value "$1" "${2:-}"
                LLM_GPU_COUNT="$2"
                shift 2
                ;;
            --output-name)
                require_model_option_value "$1" "${2:-}"
                MODEL_OUTPUT_NAME="$2"
                shift 2
                ;;
            --output-dir)
                require_model_option_value "$1" "${2:-}"
                MODEL_OUTPUT_DIR="$2"
                shift 2
                ;;
            --help|-h)
                show_model_submit_help "$wrapper_name"
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                show_model_submit_help "$wrapper_name" >&2
                exit 2
                ;;
        esac
    done

    if [[ -z "$MODEL_OUTPUT_NAME" ]]; then
        if [[ "$INSTRUCT_MODEL_ALIAS" == "$REASONING_MODEL_ALIAS" ]]; then
            MODEL_OUTPUT_NAME="$INSTRUCT_MODEL_ALIAS"
        else
            MODEL_OUTPUT_NAME="${INSTRUCT_MODEL_ALIAS}-to-${REASONING_MODEL_ALIAS}"
        fi
    fi

    if [[ ! "$MODEL_OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Invalid output name: $MODEL_OUTPUT_NAME" >&2
        echo "Use only letters, numbers, dots, underscores, and hyphens." >&2
        exit 2
    fi

    if [[ ! "$LLM_GPU_TYPE" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Invalid GPU type: $LLM_GPU_TYPE" >&2
        exit 2
    fi
    if [[ ! "$LLM_GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU count must be a positive integer: $LLM_GPU_COUNT" >&2
        exit 2
    fi

    LLM_CUDA_DEVICES=""
    local gpu_index
    for ((gpu_index = 0; gpu_index < LLM_GPU_COUNT; gpu_index++)); do
        if [[ -n "$LLM_CUDA_DEVICES" ]]; then
            LLM_CUDA_DEVICES+=","
        fi
        LLM_CUDA_DEVICES+="$gpu_index"
    done

    export INSTRUCT_MODEL_PATH REASONING_MODEL_PATH
    export INSTRUCT_MODEL_ALIAS REASONING_MODEL_ALIAS
    export MODEL_OUTPUT_NAME MODEL_OUTPUT_DIR
    export INPUT_DIR TABULAR_DIR NOTE_LINK_PATH
    export LLM_GPU_TYPE LLM_GPU_COUNT LLM_CUDA_DEVICES

    printf 'Model configuration:\n'
    printf '  instruct: %s (%s)\n' "$INSTRUCT_MODEL_PATH" "$INSTRUCT_MODEL_ALIAS"
    printf '  reasoning: %s (%s)\n' "$REASONING_MODEL_PATH" "$REASONING_MODEL_ALIAS"
    if [[ -n "$MODEL_OUTPUT_DIR" ]]; then
        printf '  output directory: %s\n' "$MODEL_OUTPUT_DIR"
    else
        printf '  output directory: PROJECT_DIR/outputs/%s\n' "$MODEL_OUTPUT_NAME"
    fi
    printf 'Source data:\n'
    printf '  notes: %s\n' "$INPUT_DIR"
    printf '  structured: %s\n' "$TABULAR_DIR"
    printf '  note link: %s\n' "$NOTE_LINK_PATH"
    printf 'LLM GPU resources:\n'
    printf '  Slurm GRES: gpu:%s:%s\n' "$LLM_GPU_TYPE" "$LLM_GPU_COUNT"
    printf '  CUDA devices: %s\n' "$LLM_CUDA_DEVICES"
}

submit_model_job() {
    local job_id

    job_id=$(sbatch --parsable --export=ALL "$@")
    printf '%s\n' "${job_id%%;*}"
}

submit_llm_job() {
    local job_id

    job_id=$(sbatch --parsable --export=ALL \
        --gres="gpu:${LLM_GPU_TYPE}:${LLM_GPU_COUNT}" "$@")
    printf '%s\n' "${job_id%%;*}"
}
