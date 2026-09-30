#!/bin/bash
#SBATCH -t 240:00:00
#
# The node is only filtered by the A30 request above: Slurm picks any free node with an A30 GPU.
# To pin a specific node, set NODE_LIST in .env and submit through the wrapper
# (`bash submit_experiment.sh`), which passes --nodelist on the sbatch command line. Slurm reads
# "#SBATCH" lines as literal text, so the node cannot be templated here.

module purge
conda activate fmperf-env
pip install -r requirements.txt

# Run your experiment
echo "Start experiment on: ${SLURM_NODELIST:-unknown} (job ${SLURM_JOB_ID:-unknown})"
python -u experiment_automation.py
