#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:45:00
# 模块大纲：所有测量/模板任务单核；数组内并发由提交命令明确限制。
set -euo pipefail
cd /pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
PY=/global/homes/l/lzy/anaconda3/envs/desilike/bin/python
case "$1" in
  geometry) "$PY" -B code/run_rsd_bonly_workflow.py geometry --meshsize "${2:-128}" ;;
  pilot) start=$((SLURM_ARRAY_TASK_ID*10)); "$PY" -B code/run_rsd_bonly_workflow.py measure --pilot --start "$start" --stop "$((start+10))" ;;
  fine) start=$((SLURM_ARRAY_TASK_ID*100)); "$PY" -B code/run_rsd_bonly_workflow.py measure --role validation --meshsize 256 --start "$start" --stop "$((start+1))" ;;
  bulk) start=$((SLURM_ARRAY_TASK_ID*20)); "$PY" -B code/run_rsd_bonly_workflow.py measure --start "$start" --stop "$((start+20))" ;;
  templates) start=$((SLURM_ARRAY_TASK_ID*2)); "$PY" -B code/run_rsd_bonly_workflow.py templates --start "$start" --stop "$((start+2))" ;;
  aggregate) "$PY" -B code/run_rsd_bonly_workflow.py aggregate ;;
  mcmc) "$PY" -B code/run_rsd_bonly_mcmc.py --case-index "$SLURM_ARRAY_TASK_ID" ;;
  *) echo "unknown job"; exit 2 ;;
esac
