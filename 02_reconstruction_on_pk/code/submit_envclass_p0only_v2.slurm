#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -J envclass_p0
#SBATCH --array=0-599%4
#SBATCH -o /pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/envclass_p0only/logs/task_%a.out
#SBATCH -e /pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/envclass_p0only/logs/task_%a.err

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

CODE=/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/code/run_envclass_p0_only.py

if [ $SLURM_ARRAY_TASK_ID -lt 500 ]; then
    COSMO=fiducial
    REAL=$SLURM_ARRAY_TASK_ID
else
    COSMO=LC_p
    REAL=$((SLURM_ARRAY_TASK_ID - 500))
fi

# 已完成则跳过（可安全重投）。
DONE=/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/envclass_p0only/full/$COSMO/real$(printf '%03d' $REAL)/DONE
if [ -f "$DONE" ]; then
    echo "skip $COSMO real$REAL (DONE exists)"
    exit 0
fi

srun python $CODE --cosmology $COSMO --realization $REAL
