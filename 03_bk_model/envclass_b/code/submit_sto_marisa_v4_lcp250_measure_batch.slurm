#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --job-name=smv4_pb_batch

# 每个 allocation 顺序测量一个 manifest 分块，避免每套 P+B 之间重复排队。
set -eo pipefail
ROOT=/pscratch/sd/l/lzy/reconstuction-png-v2
WORK="${ROOT}/03_bk_model/envclass_b/work/sto_marisa_v4_lcp250"
MANIFEST="${WORK}/measurement_tail_real100_249.jsonl"
RESULT_ROOT="${ROOT}/01_reconstruction_methods/outputs/sto_marisa_v4_lcp250/full/LC_p"
CHUNK="${CHUNK:-10}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?需要 Slurm array task id}"
START=$((ARRAY_INDEX * CHUNK))
N_TASKS=$(wc -l < "${MANIFEST}")
END=$((START + CHUNK - 1))
if [[ "${END}" -ge "${N_TASKS}" ]]; then
    END=$((N_TASKS - 1))
fi

source "${ROOT}/01_reconstruction_methods/common/nersc_env.sh"
cd "${ROOT}"
echo "[$(date -u)] measurement batch rows ${START}..${END} / $((N_TASKS - 1))"
for INDEX in $(seq "${START}" "${END}"); do
    REAL=$((INDEX + 100))
    DONE="${RESULT_ROOT}/real$(printf '%03d' "${REAL}")/DONE"
    if [[ -f "${DONE}" ]]; then
        echo "[$(date -u)] skip LC_p real${REAL}: DONE exists"
        continue
    fi
    echo "[$(date -u)] measurement task-index ${INDEX} (real${REAL})"
    srun --ntasks=1 --cpus-per-task=8 --kill-on-bad-exit=1 \
        python 01_reconstruction_methods/combined/run_sto_marisa.py \
        --manifest "${MANIFEST}" --task-index "${INDEX}"
done
echo "[$(date -u)] measurement batch complete"
