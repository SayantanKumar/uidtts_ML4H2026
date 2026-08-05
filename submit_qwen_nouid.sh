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

j12=$(sbatch --parsable step_12_uid_tagging.sh)
j12=${j12%%;*}

j34=$(sbatch --parsable --dependency=afterok:"$j12" step_34_temporal.sh)
j34=${j34%%;*}

j5a=$(sbatch --parsable --dependency=afterok:"$j34" step_5a.sh)
j5a=${j5a%%;*}

j5b=$(sbatch --parsable --dependency=afterok:"$j5a" step_5b.sh)
j5b=${j5b%%;*}

j5c=$(sbatch --parsable --dependency=afterok:"$j5b" step_5c.sh)
j5c=${j5c%%;*}

j5d=$(sbatch --parsable --dependency=afterok:"$j5c" step_5d.sh)
j5d=${j5d%%;*}

j5e=$(sbatch --parsable --dependency=afterok:"$j5d" step_5e.sh)
j5e=${j5e%%;*}

j6=$(sbatch --parsable --dependency=afterok:"$j5e" step_6_nouid.sh)
j6=${j6%%;*}

jpost=$(sbatch --parsable --dependency=afterok:"$j6" post_processing_nouid.sh)
jpost=${jpost%%;*}

printf 'Submitted dependency chain:\n'
printf '12=%s → 34=%s → 5a=%s → 5b=%s → 5c=%s → 5d=%s → 5e=%s → 6=%s → post=%s\n' \
    "$j12" "$j34" "$j5a" "$j5b" "$j5c" "$j5d" "$j5e" "$j6" "$jpost"