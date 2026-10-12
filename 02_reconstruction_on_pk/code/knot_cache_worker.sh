#!/bin/bash
# 用法: knot_cache_worker.sh <cosmo> <start> <end> —— knot 裸缓存 [start,end)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/code
for ((R=$2; R<$3; R++)); do
  python3 cache_one_knot.py $R $1 >> ../envclass_p0only/logs/knot_cache_$1.log 2>&1
done
echo "knot cache $1 [$2,$3) done"
