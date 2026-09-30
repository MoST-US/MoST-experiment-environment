#!/bin/bash
set -euo pipefail

# Move to the script's directory
cd "$(dirname "$0")"

# Load variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Pass --nodelist if NODE_LIST is defined
EXTRA_ARGS=()
if [ -n "${NODE_LIST:-}" ]; then
    EXTRA_ARGS+=("--nodelist=$NODE_LIST")
fi

echo "Submitting job with NODE_LIST=${NODE_LIST:-<unset>}"
exec sbatch "${EXTRA_ARGS[@]}" "$@" experiment_automation.slurm