#!/bin/bash
# Submit experiment_automation.py to Slurm, optionally pinning the node that runs it.
#
# The node must be chosen at SUBMISSION time: Slurm reads "#SBATCH" lines as literal text
# (shell syntax and variables are not expanded), so --nodelist cannot be a directive inside
# experiment_automation.slurm and .env is never read by Slurm itself. This wrapper resolves
# NODE_LIST and passes --nodelist on the sbatch command line instead.
#
#   bash submit_experiment.sh                  # NODE_LIST from the environment, else from .env
#   NODE_LIST=gpu07 bash submit_experiment.sh  # one-off override
#   bash submit_experiment.sh --partition=p    # extra sbatch options are passed through
#
# With NODE_LIST unset the submission is exactly the old one and Slurm picks any free A30 node
# (see the #SBATCH --gpus=a30:1 directive in experiment_automation.slurm).
set -euo pipefail

cd "$(dirname "$0")"

ENV_FILE=.env
JOB_SCRIPT=experiment_automation.slurm

# Read one KEY=VALUE from .env without modifying it. This mirrors the tolerant reader duplicated in
# the Python tools: CRLF endings, blank lines, comments, "export " prefixes and spaces around "=" are
# accepted, the value is trimmed and surrounding quotes are removed (an inline comment is NOT
# stripped, exactly like `_read_env_value` in experiment_automation.py).
read_env_value() {
    local key="$1"
    local line=""
    local name=""
    local value=""

    [ -f "$ENV_FILE" ] || return 0

    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line#export }"
        line="${line#"${line%%[![:space:]]*}"}"
        case "$line" in
            ''|'#'*) continue ;;
        esac
        case "$line" in
            *=*) ;;
            *) continue ;;
        esac
        name="${line%%=*}"
        name="${name%"${name##*[![:space:]]}"}"
        if [ "$name" = "$key" ]; then
            value="${line#*=}"
            break
        fi
    done < "$ENV_FILE"

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    printf '%s' "$value"
}

NODE_LIST_VALUE=''
NODE_LIST_SOURCE='unset'
if [ -n "${NODE_LIST:-}" ]; then
    NODE_LIST_VALUE="$NODE_LIST"
    NODE_LIST_SOURCE='environment'
else
    NODE_LIST_VALUE="$(read_env_value NODE_LIST)"
    if [ -n "$NODE_LIST_VALUE" ]; then
        NODE_LIST_SOURCE="$ENV_FILE"
    fi
fi

# Never abort on a suspicious value: sbatch validates the nodelist itself.
case "$NODE_LIST_VALUE" in
    *[[:space:]=]*)
        echo "Warning: NODE_LIST contains an unexpected character: '$NODE_LIST_VALUE'"
        ;;
esac

NODELIST_FROM_ARGS=0
for arg in "$@"; do
    case "$arg" in
        -w|--nodelist|--nodelist=*) NODELIST_FROM_ARGS=1 ;;
    esac
done

if [ -n "$NODE_LIST_VALUE" ]; then
    if [ "$NODELIST_FROM_ARGS" -eq 1 ]; then
        echo "Note: --nodelist was already given on the command line; ignoring NODE_LIST ($NODE_LIST_VALUE)."
    else
        set -- "$@" "--nodelist=$NODE_LIST_VALUE"
    fi
fi

# Best-effort sanity check: a plainly named node that scontrol does not know would be rejected by
# sbatch anyway, so this is a warning, never a failure (nodelist patterns such as 'gpu[05-07]' are
# skipped). It passes the value as a whole, so comma-separated lists are accepted by scontrol too.
if [ -n "$NODE_LIST_VALUE" ] && command -v scontrol >/dev/null 2>&1 \
        && [ "${NODE_LIST_VALUE#*\[}" = "$NODE_LIST_VALUE" ]; then
    if ! scontrol show node "$NODE_LIST_VALUE" >/dev/null 2>&1; then
        echo "Warning: scontrol does not know node '$NODE_LIST_VALUE'; the job is rejected if the name is invalid."
    fi
fi

echo "NODE_LIST: ${NODE_LIST_VALUE:-<unset>} (source: $NODE_LIST_SOURCE)"
echo "Submitting: sbatch $* $JOB_SCRIPT"
exec sbatch "$@" "$JOB_SCRIPT"
