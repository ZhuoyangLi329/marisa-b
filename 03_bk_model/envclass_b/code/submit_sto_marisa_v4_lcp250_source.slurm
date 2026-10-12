#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=smv4_lcp_src

set -eo pipefail
ROOT=/pscratch/sd/l/lzy/reconstuction-png-v2
WORK="${ROOT}/03_bk_model/envclass_b/work/sto_marisa_v4_lcp250"
MANIFEST="${WORK}/source_tail_real100_249.jsonl"
INDEX="${SLURM_ARRAY_TASK_ID:?需要 Slurm array task id}"

source "${ROOT}/01_reconstruction_methods/common/nersc_env.sh"
cd "${ROOT}"
srun --ntasks=1 --cpus-per-task=8 --kill-on-bad-exit=1 \
    python 01_reconstruction_methods/sto_recon/run_halo_sto_variants.py \
    --manifest "${MANIFEST}" --task-index "${INDEX}" --skip-existing
