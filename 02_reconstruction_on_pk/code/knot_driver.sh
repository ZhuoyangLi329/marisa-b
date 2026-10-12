#!/bin/bash
# knot-only 补产：fid 100 + LC_p 100（供校准）。
W=$1; N=$2; COSMO=$3; NREAL=$4
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/code
for ((R=W; R<NREAL; R+=N)); do
    D=../envclass_p0only/full/$COSMO/real$(printf '%03d' $R)/DONE
    if [ -f "$D" ]; then
        # DONE 存在但 knot_pk 可能缺失（当时只测三类）
        if [ -f ../envclass_p0only/full/$COSMO/real$(printf '%03d' $R)/knot_pk.npz ]; then continue; fi
    fi
    python run_envclass_p0_only.py --cosmology $COSMO --realization $R --classes knot         >> ../envclass_p0only/logs/knot_${COSMO}_w${W}.log 2>&1
done
echo "knot worker $W ($COSMO) done"
