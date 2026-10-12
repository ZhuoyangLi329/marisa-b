#!/bin/bash
# 互谱生产驱动：vs/vf/sf 三对，fid 500 + LC_p 100。
W=$1; N=$2; COSMO=$3; NREAL=$4
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/code
for ((R=W; R<NREAL; R+=N)); do
    if [ -f ../envclass_p0only/full/$COSMO/real$(printf '%03d' $R)/cross_voidsheet_pk.npz ]; then continue; fi
    python run_envclass_crosspk.py --cosmology $COSMO --realization $R --pairs vs,vf,sf,vk,sk,fk \
        >> ../envclass_p0only/logs/cross_${COSMO}_w${W}.log 2>&1
done
echo "cross worker $W ($COSMO) done"
