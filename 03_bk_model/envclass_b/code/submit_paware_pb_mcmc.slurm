#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=paware_pb_mcmc

# 四个独立 case 各占一个 CPU；不启动任何数据重产。
set -eo pipefail
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
TASK_ROOT=/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b
SCRIPT="$TASK_ROOT/code/run_paware_pb_mcmc.py"
OUTPUT="$TASK_ROOT/outputs/paware_pb_mcmc_k012_k014_v1"
PIDS=()
for CASE in pre_k0p12 pre_k0p14 sto_marisa_k0p12 sto_marisa_k0p14; do
    python -u "$SCRIPT" sample --case "$CASE" > "$OUTPUT/$CASE/sampling.log" 2>&1 &
    PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=1
done
exit "$FAILED"
