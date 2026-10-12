#!/bin/bash
# Phase 1c worker：sto_marisa LC_p v2 重产 [start,end) task-index 分片。
# 用法: p1_worker.sh <start> <end>
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
cd /pscratch/sd/l/lzy/reconstruction-png
MANIFEST=/pscratch/sd/l/lzy/reconstruction-png/manifests_v2_paware/quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100_sto_marisa_b000_pk_v2_p1282/tasks_all.jsonl
mkdir -p /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/work/logs_p1
for ((I=$1; I<$2; I++)); do
  TAG=$(printf "%03d" $I)
  DONEF=/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100_sto_marisa_b000_pk_v2_p1282/full/LC_p/real$TAG/DONE
  if [ -f "$DONEF" ]; then continue; fi
  python3 codes/run_quijote_halo_z0p5_jaxrecon_jaxpower_b000.py \
    --manifest $MANIFEST --task-index $I \
    >> /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/work/logs_p1/worker_$1.log 2>&1
done
echo "p1 [$1,$2) done"
