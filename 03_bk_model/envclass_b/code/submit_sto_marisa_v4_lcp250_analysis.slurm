#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=smv4_pb_fit

set -eo pipefail
ROOT=/pscratch/sd/l/lzy/reconstuction-png-v2
source "${ROOT}/01_reconstruction_methods/common/nersc_env.sh"
cd "${ROOT}"
python 03_bk_model/envclass_b/code/run_joint_pb_matched250.py
