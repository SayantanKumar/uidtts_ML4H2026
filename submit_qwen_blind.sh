#!/bin/bash
set -euo pipefail

# Run with:
#   bash submit_qwen_nouid.sh

WRAPPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$WRAPPER_DIR/run_scripts_qwen"

[[ -d "$SCRIPT_DIR" ]] || {
    echo "Directory not found: $SCRIPT_DIR" >&2
    exit 1
}

cd "$SCRIPT_DIR"
mkdir -p sbatch_output sbatch_error

j6=$(sbatch --parsable step_6_blind.sh)
j6=${j6%%;*}

jpost=$(sbatch --parsable --dependency=afterok:"$j6" post_processing_blind.sh)
jpost=${jpost%%;*}

printf 'Submitted dependency chain:\n'
printf '6=%s → post=%s\n' \
    "$j6" "$jpost"