#!/bin/bash
# 登录节点多路并行驱动：fiducial 与 LC_p 各 N 路，总核数 <=12。
# 用法：parallel_driver.sh <worker_id> <n_workers> <cosmology> <n_real>
# 每个 worker 处理 realization ≡ worker_id (mod n_workers) 的子集。

W=$1; N=$2; COSMO=$3; NREAL=$4

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/code

for ((R=W; R<NREAL; R+=N)); do
    D=../envclass_p0only/full/$COSMO/real$(printf '%03d' $R)/DONE
    if [ -f "$D" ]; then continue; fi
    python run_envclass_p0_only.py --cosmology $COSMO --realization $R \
        >> ../envclass_p0only/logs/par_${COSMO}_w${W}.log 2>&1
done
echo "worker $W ($COSMO) finished"
