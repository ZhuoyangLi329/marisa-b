#!/usr/bin/env bash
#SBATCH --job-name=v2_marisa_gra
#SBATCH --account=desi_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
# 示例只含一条；申请 32 CPU 是现有 shared GPU 模板，库线程固定为 8。
# 提交时用 --chdir、--output、--error 指向本部分，详见 README。
set -eo pipefail
PART_DIR="${PART_DIR:-/pscratch/sd/l/lzy/reconstuction-png-v2/01_reconstruction_methods}"
source "${PART_DIR}/common/nersc_env.sh"
cd "${PART_DIR}"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 --gpu-bind=single:1 \
    python "${PART_DIR}/marisa_gra/run_halo_b000_marisa_gra.py" \
    --manifest "${MANIFEST_PATH:-${PART_DIR}/examples/marisa_gra.jsonl}" \
    --task-index "${SLURM_ARRAY_TASK_ID:-0}" --stage "${STAGE:-full}"
