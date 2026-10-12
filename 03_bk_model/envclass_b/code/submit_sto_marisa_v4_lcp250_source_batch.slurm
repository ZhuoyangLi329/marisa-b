#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=smv4_src_batch

# 把短 source 任务分块顺序执行，减少 shared 队列对 150 个独立 array task 的调度开销。
set -eo pipefail
ROOT=/pscratch/sd/l/lzy/reconstuction-png-v2
WORK="${ROOT}/03_bk_model/envclass_b/work/sto_marisa_v4_lcp250"
MANIFEST="${WORK}/source_tail_real100_249.jsonl"
CHUNK="${CHUNK:-19}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?需要 Slurm array task id}"
START=$((ARRAY_INDEX * CHUNK))
N_TASKS=$(wc -l < "${MANIFEST}")
END=$((START + CHUNK - 1))
if [[ "${END}" -ge "${N_TASKS}" ]]; then
    END=$((N_TASKS - 1))
fi

source "${ROOT}/01_reconstruction_methods/common/nersc_env.sh"
cd "${ROOT}"
echo "[$(date -u)] source batch rows ${START}..${END} / $((N_TASKS - 1))"
for INDEX in $(seq "${START}" "${END}"); do
    echo "[$(date -u)] source task-index ${INDEX}"
    srun --ntasks=1 --cpus-per-task=8 --kill-on-bad-exit=1 \
        python 01_reconstruction_methods/sto_recon/run_halo_sto_variants.py \
        --manifest "${MANIFEST}" --task-index "${INDEX}" --skip-existing
done
echo "[$(date -u)] source batch complete"
