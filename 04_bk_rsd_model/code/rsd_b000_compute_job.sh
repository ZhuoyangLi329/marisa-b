#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --job-name=rsd_B000_theory
# 模块大纲：模板/审计/MCMC 共用轻量单核 desilike 环境。
set -euo pipefail
cd /pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/global/homes/l/lzy/anaconda3/envs/desilike/bin/python
case "$1" in
  template) "$PY" -B -u code/build_rsd_b000_templates.py --index "$SLURM_ARRAY_TASK_ID" --probe ;;
  template_fine) "$PY" -B -u code/build_rsd_b000_templates.py --index "$SLURM_ARRAY_TASK_ID" --meshsize 160 --probe ;;
  pilot_audit) "$PY" -B -u code/run_rsd_b000_inference.py pilot ;;
  aggregate) "$PY" -B -u code/run_rsd_b000_inference.py aggregate ;;
  correct_covariance) "$PY" -B -u code/run_rsd_b000_inference.py correct_covariance ;;
  workflow_audit) "$PY" -B -u code/audit_rsd_b000_workflow.py ;;
  mcmc) "$PY" -B -u code/run_rsd_b000_mcmc.py --case-index "$SLURM_ARRAY_TASK_ID" ;;
  sensitivity) "$PY" -B -u code/run_rsd_b000_inference.py sensitivity ;;
  report) "$PY" -B -u code/report_rsd_b000.py --require-complete
          "$PY" -B -u code/analyze_rsd_b000_results.py ;;
  *) echo unknown_compute_job; exit 2 ;;
esac
