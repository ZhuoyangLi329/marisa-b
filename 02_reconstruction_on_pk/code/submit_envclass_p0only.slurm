#!/bin/bash
#SBATCH --job-name=envclass_p0
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --partition=regular_milan_ss11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --array=0-599%12
#SBATCH --output=/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/envclass_p0only/logs/task_%a.out
#SBATCH --error=/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/envclass_p0only/logs/task_%a.err

# 环境类 P0-only 生产：array 0-599 = fiducial 0-499 + LC_p 0-99。
# 每任务测 void/sheet/filament 三类 P0（自建几何 v1）。

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
