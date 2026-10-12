#!/bin/bash
# 用法: cache_worker.sh <start> <end> <cosmo> —— 处理 [start,end)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/code
for ((R=$1; R<$2; R++)); do
    python3 cache_one.py $R $3 >> ../envclass_p0only/logs/cache_$3.log 2>&1
done
echo "cache $3 [$1,$2) done"
