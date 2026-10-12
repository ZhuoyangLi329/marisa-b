#!/bin/bash
# 登录节点 worker（抗抖版）：marisa only v2 重产，task-index [start,end)。
# 不 cd 旧仓库（Lustre 抖动会让 cd 失败）；runner 用绝对路径；python 失败重试。
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
RUNNER=/pscratch/sd/l/lzy/reconstruction-png/codes/run_quijote_halo_z0p5_jaxrecon_jaxpower_b000.py
MANIFEST=/pscratch/sd/l/lzy/reconstruction-png/manifests_v2_paware/quijote_halo_z0p5_r10_stovar_unit_fid500_lcp100_marisa_b000_pk_v2_p1217/tasks_all.jsonl
ROOT=/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_stovar_unit_fid500_lcp100_marisa_b000_pk_v2_p1217
LOGDIR=/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/work/logs_p2
mkdir -p "$LOGDIR"
for ((I=$1; I<$2; I++)); do
  if [ $I -lt 500 ]; then COSMO=fiducial; REAL=$I; else COSMO=LC_p; REAL=$((I-500)); fi
  TAG=$(printf "%03d" $REAL)
  if [ -f "$ROOT/full/$COSMO/real$TAG/DONE" ]; then continue; fi
  for attempt in 1 2 3; do
    python3 "$RUNNER" --manifest "$MANIFEST" --task-index $I >> "$LOGDIR/worker_$1.log" 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then break; fi
    echo "task $I attempt $attempt failed rc=$RC, retrying" >> "$LOGDIR/worker_$1.log"
    sleep 30
  done
done
echo "p2 [$1,$2) done"
