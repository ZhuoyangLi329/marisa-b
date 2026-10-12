#!/bin/bash
# 用法: knot_bare_worker.sh <cosmo> <start> <end>
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/code
for ((R=$2; R<$3; R++)); do
  python3 knot_bare_one.py $R $1 >> ../envclass_p0only/logs/knot_bare_$1.log 2>&1
done
echo "knot bare $1 [$2,$3) done"
