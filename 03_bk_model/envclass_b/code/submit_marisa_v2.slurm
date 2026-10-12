#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -J marisa_v2p1217
#SBATCH --array=0-599%8
#SBATCH -o /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/work/logs_p2/slurm_%a.out
#SBATCH -e /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/work/logs_p2/slurm_%a.err

# Phase 2+4：marisa only v2 重产（png_p=1.1217、bias=1.9364）600 套
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

CODE=/pscratch/sd/l/lzy/reconstruction-png/codes/run_quijote_halo_z0p5_jaxrecon_jaxpower_b000.py
MANIFEST=/pscratch/sd/l/lzy/reconstruction-png/manifests_v2_paware/quijote_halo_z0p5_r10_stovar_unit_fid500_lcp100_marisa_b000_pk_v2_p1217/tasks_all.jsonl
ROOT=/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_stovar_unit_fid500_lcp100_marisa_b000_pk_v2_p1217

IDX=$SLURM_ARRAY_TASK_ID
if [ $IDX -lt 500 ]; then COSMO=fiducial; REAL=$IDX; else COSMO=LC_p; REAL=$((IDX-500)); fi
DONE=$ROOT/full/$COSMO/real$(printf '%03d' $REAL)/DONE
if [ -f "$DONE" ]; then
    echo "skip $COSMO real$REAL (DONE exists)"
    exit 0
fi
srun python3 $CODE --manifest $MANIFEST --task-index $IDX
