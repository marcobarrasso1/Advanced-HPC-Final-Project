#!/bin/bash
#SBATCH --job-name=jacobi-scale
#SBATCH -A ICT25_MHPC_0
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=00:50:00
#SBATCH -o jacobi.%j.out
#SBATCH -e jacobi.%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load nvhpc/24.3 openmpi/4.1.6--nvhpc--24.3

# OpenMP settings (host threads)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_TARGET_OFFLOAD=MANDATORY

# Problem size
N=10000
ITERS=1000

# Sweep values
TEAMS_LIST=(108 432 864 1728 2592 3456)
TLIST=(128 256 512 1024 2048)


echo "Allocated nodes:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

MAX_NODES=${SLURM_JOB_NUM_NODES:-1}
nodes=()
for ((i=1; i<=MAX_NODES; i++)); do
  nodes+=("$i")
done


CSV="info.csv"
echo "nodes,total_ranks,teams,thread_limit,N,ITERS,reps,median_s,mean_s,min_s,max_s" > "$CSV"

REPS=10
WARMUP=3

extract_time() {
  # Extract the last occurrence of "TIME: <number>"
  awk '/^TIME:[[:space:]]*/ {t=$2} END { if (t=="") exit 2; print t }'
}

for NODES in "${nodes[@]}"; do
  RPN=${SLURM_NTASKS_PER_NODE:-4}
  TOTAL_RANKS=$((NODES * RPN))

  for num_teams in "${TEAMS_LIST[@]}"; do
    for thread_limit in "${TLIST[@]}"; do

      echo "--- nodes=${NODES} ranks=${TOTAL_RANKS} teams=${num_teams} tl=${thread_limit} ---"

      # Warmup (discard output)
      for ((w=1; w<=WARMUP; w++)); do
        srun --nodes "$NODES" \
             --ntasks-per-node "$RPN" \
             --cpus-per-task "$SLURM_CPUS_PER_TASK" \
             --gpus-per-task 1 \
             --gpu-bind=closest \
             --cpu-bind=cores \
             ./jacobi3.x "$N" "$ITERS" "$SLURM_CPUS_PER_TASK" "$num_teams" "$thread_limit" \
             >/dev/null 2>&1
      done

      times=()
      for ((r=1; r<=REPS; r++)); do
        out="$(
          srun --nodes "$NODES" \
               --ntasks-per-node "$RPN" \
               --cpus-per-task "$SLURM_CPUS_PER_TASK" \
               --gpus-per-task 1 \
               --gpu-bind=closest \
               --cpu-bind=cores \
               ./jacobi3.x "$N" "$ITERS" "$SLURM_CPUS_PER_TASK" "$num_teams" "$thread_limit" \
               2>&1
        )"

        # Parse TIME
        if ! t="$(printf '%s\n' "$out" | extract_time)"; then
          echo "ERROR: could not find TIME line. Full output was:"
          echo "-----"
          echo "$out"
          echo "-----"
          exit 1
        fi

        times+=("$t")
      done

      stats="$(python3 - <<'PY' "${times[@]}"
import sys, statistics
xs = [float(x) for x in sys.argv[1:]]
xs.sort()
median = statistics.median(xs)
mean = sum(xs)/len(xs)
mn, mx = xs[0], xs[-1]
print(f"{median:.9f},{mean:.9f},{mn:.9f},{mx:.9f}")
PY
)"
      median="${stats%%,*}"
      rest="${stats#*,}"
      mean="${rest%%,*}"
      rest="${rest#*,}"
      mn="${rest%%,*}"
      mx="${rest#*,}"

      echo "${NODES},${TOTAL_RANKS},${num_teams},${thread_limit},${N},${ITERS},${REPS},${median},${mean},${mn},${mx}" >> "$CSV"
      echo "   median=${median} mean=${mean}"
    done
  done
done
