#!/bin/bash
#SBATCH --job-name=mimo_charpos
#SBATCH --partition=gpu
#SBATCH --constraint=gpuh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Submit with:
#   sbatch run_scripts_slurm/repair_char_positions_mimo_h200.sh
# Override any case path without editing this file, for example:
#   TIMELINE_FILE=/data/.../case.bsv NOTE_FILE=/data/.../case.txt \
#   OUTPUT_FILE=/data/.../case_repaired.bsv \
#   sbatch run_scripts_slurm/repair_char_positions_mimo_h200.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/grimalditersans/uid_anchor_detection}"
MODEL_DIR="${MODEL_DIR:-/data/grimalditersans/models/MiMo-V2.6-Pro-RL}"
TIMELINE_FILE="${TIMELINE_FILE:-/data/grimalditersans/MLHC_TTA/compare_tts/m4rnd20_batch_output_0001/bundle/charpos/11649157-DS-7_positions_timeline_1.bsv}"
NOTE_FILE="${NOTE_FILE:-/data/grimalditersans/mimic_uid_data/i2m4_n40_notes/11649157.txt}"
OUTPUT_FILE="${OUTPUT_FILE:-/data/grimalditersans/MLHC_TTA/compare_tts/m4rnd20_batch_output_0001/bundle/charpos_llm/11649157-DS-7_positions_timeline_1.bsv}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---dtype auto}"

source /data/grimalditersans/myconda
conda activate /data/grimalditersans/mymamba/envs/gen-vllm

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

for required_path in "$REPO_DIR/repair_char_positions_llm.py" "$MODEL_DIR" "$TIMELINE_FILE" "$NOTE_FILE"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required path does not exist: $required_path" >&2
        exit 1
    fi
done

if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vllm is unavailable after activating the environment" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Node:       $(hostname -s)"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Model:      $MODEL_DIR"
echo "Timeline:   $TIMELINE_FILE"
echo "Note:       $NOTE_FILE"
echo "Output:     $OUTPUT_FILE"

command=(
    python3 "$REPO_DIR/repair_char_positions_llm.py"
    --timeline "$TIMELINE_FILE"
    --note "$NOTE_FILE"
    --output "$OUTPUT_FILE"
    --model-path "$MODEL_DIR"
    --tensor-parallel-size 8
    --startup-timeout 3600
    --repair-all
    --overwrite
)
if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
    command+=(--vllm-extra-args "$VLLM_EXTRA_ARGS")
fi

set +e
"${command[@]}"
status=$?
set -e

# The Python tool returns 1 when it safely writes an output containing some
# unresolved rows. Preserve that useful result without marking the Slurm job
# failed; genuine startup/argument failures do not produce the output table.
if [[ "$status" -eq 1 && -s "$OUTPUT_FILE" ]]; then
    echo "Completed with unresolved rows; inspect ${OUTPUT_FILE}.audit.jsonl" >&2
    exit 0
fi
exit "$status"
