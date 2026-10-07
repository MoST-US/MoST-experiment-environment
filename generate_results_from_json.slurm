#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodelist=c10
#
# Generate the derived results_from_json.csv of every iteration under RESULTS_DIR that is
# missing it. This is a CPU/IO-only pass: it needs no GPU and no inference endpoint. Pass
# a results root as an argument to override the default (RESULTS_DIR from the environment
# or .env), e.g.: sbatch generate_results_from_json.slurm results

module purge
conda activate fmperf-env
pip install -r requirements.txt

echo "Generating results_from_json.csv on: ${SLURM_NODELIST:-unknown} (job ${SLURM_JOB_ID:-unknown})"
python -u requests/generate_results_from_json.py "$@"
