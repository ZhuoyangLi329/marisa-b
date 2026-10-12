#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --job-name=rsd_B000
# 模块大纲：计算节点复用 cosmodesi，不安装依赖；一个 rank、B-only。
set -eo pipefail
cd /pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
source ../01_reconstruction_methods/common/nersc_env.sh
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
case "$1" in
  prototype) python -B -u code/measure_rsd_b000.py --start 0 --stop 1 --control-realspace ;;
  pilot) start=$((SLURM_ARRAY_TASK_ID*10)); python -B -u code/measure_rsd_b000.py --pilot --start "$start" --stop "$((start+10))" --control-realspace ;;
  fine) start=$((SLURM_ARRAY_TASK_ID*100)); python -B -u code/measure_rsd_b000.py --start "$start" --stop "$((start+1))" --meshsize 512 ;;
  bulk) start=$((SLURM_ARRAY_TASK_ID*10)); python -B -u code/measure_rsd_b000.py --start "$start" --stop "$((start+10))" ;;
  *) echo unknown_measurement_job; exit 2 ;;
esac
