#!/bin/bash
#SBATCH --job-name=jacobi-scale
#SBATCH -A ICT25_MHPC_0
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=64                 # max nodes you plan to test
#SBATCH --ntasks-per-node=4        # 1 MPI rank per GPU
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH -o jacobi.%j.out
#SBATCH -e jacobi.%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load nvhpc/24.3 openmpi/4.1.6--nvhpc--24.3

# OpenMP settings
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_TARGET_OFFLOAD=MANDATORY

# Problem size
N=10000
ITERS=1000
nodes=(1 2 4 8 16 24 32 48 64)


echo "Allocated nodes:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

for NODES in "${nodes[@]}"; do
  echo "=== RUN with NODES=${NODES} (ranks = ${NODES} * ${SLURM_NTASKS_PER_NODE}) ==="

  srun --nodes "$NODES" \
       --ntasks-per-node 4 \
       --cpus-per-task "$SLURM_CPUS_PER_TASK" \
       --gpus-per-task 1 \
       --gpu-bind=closest \
       --cpu-bind=cores \
       --exclusive \
       ./jacobi "$N" "$ITERS" "$SLURM_CPUS_PER_TASK" 108 128 
done

