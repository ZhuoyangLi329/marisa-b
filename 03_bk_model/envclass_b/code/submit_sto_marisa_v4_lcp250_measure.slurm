#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=smv4_lcp_pb

set -eo pipefail
ROOT=/pscratch/sd/l/lzy/reconstuction-png-v2
WORK="${ROOT}/03_bk_model/envclass_b/work/sto_marisa_v4_lcp250"
MANIFEST="${WORK}/measurement_tail_real100_249.jsonl"
INDEX="${SLURM_ARRAY_TASK_ID:?需要 Slurm array task id}"
REAL=$((INDEX + 100))
DONE="${ROOT}/01_reconstruction_methods/outputs/sto_marisa_v4_lcp250/full/LC_p/real$(printf '%03d' "${REAL}")/DONE"

if [[ -f "${DONE}" ]]; then
    echo "skip LC_p real${REAL}: DONE exists"
    exit 0
fi
source "${ROOT}/01_reconstruction_methods/common/nersc_env.sh"
cd "${ROOT}"
srun --ntasks=1 --cpus-per-task=8 --kill-on-bad-exit=1 \
    python 01_reconstruction_methods/combined/run_sto_marisa.py \
    --manifest "${MANIFEST}" --task-index "${INDEX}"
