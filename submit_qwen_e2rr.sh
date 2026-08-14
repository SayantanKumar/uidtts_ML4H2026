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

j5d=$(sbatch --parsable step_5d_e2rr.sh)
j5d=${j5d%%;*}

j5e=$(sbatch --parsable --dependency=afterok:"$j5d" step_5e_e2rr.sh)
j5e=${j5e%%;*}

j6=$(sbatch --parsable --dependency=afterok:"$j5e" step_6_e2rr.sh)
j6=${j6%%;*}

jpost=$(sbatch --parsable --dependency=afterok:"$j6" post_processing_e2rr.sh)
jpost=${jpost%%;*}

printf 'Submitted dependency chain:\n'
printf '5d=%s → 5e=%s → 6=%s → post=%s\n' \
    "$j5d" "$j5e" "$j6" "$jpost"