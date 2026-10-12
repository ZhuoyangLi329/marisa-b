#!/bin/bash
# 用法: crosspk_worker.sh <cosmo> <start> <end> —— 补产 sk,fk 互谱 [start,end)
# 注意：与已有 vs/vf/sf 互谱同一版本 run_envclass_crosspk.py（含 +0.5 平移），
# 保证 sk/fk 与既有互谱口径一致。
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/code
for ((R=$2; R<$3; R++)); do
  TAG=$(printf "%03d" $R)
  A=../envclass_p0only/full/$1/real$TAG/cross_sheetknot_pk.npz
  B=../envclass_p0only/full/$1/real$TAG/cross_filamentknot_pk.npz
  if [ ! -f "$A" ] || [ ! -f "$B" ]; then
    python3 run_envclass_crosspk.py --cosmology $1 --realization $R --pairs sk,fk \
      >> ../envclass_p0only/logs/crosspk_repair_$1.log 2>&1
  fi
done
echo "crosspk $1 [$2,$3) done"
