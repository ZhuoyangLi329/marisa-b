#!/bin/bash
# 用法: envclass_b_worker.sh <cosmo> <start> <end> —— 环境类 B000 [start,end)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
cd /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/code
mkdir -p ../work
for ((R=$2; R<$3; R++)); do
  TAG=$(printf "%03d" $R)
  if [ ! -f "../work/$1/real$TAG/sheet_b000.npz" ] || [ ! -f "../work/$1/real$TAG/filament_b000.npz" ]; then
    python3 run_envclass_b000.py --cosmology $1 --realization $R >> ../work/logs/worker_$1_$2.log 2>&1
  fi
done
echo "envclassB $1 [$2,$3) done"
