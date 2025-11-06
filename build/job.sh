#!/bin/bash
#SBATCH --job-name=estimator
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


OUTFILE="result.txt"

# ---- PRAS ----
EXE="./add_csv"
DATA_CSV="./c4a1.csv"
CHUNK_LEN=5000
OVERLAP=300
K=1000

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores


if [ ! -x "$EXE" ]; then
  echo "ERROR: EXE not found or not executable: $EXE" >&2
  exit 2
fi
if [ ! -f "$DATA_CSV" ]; then
  echo "ERROR: DATA_CSV not found: $DATA_CSV" >&2
  exit 2
fi

echo "[$(date)] launching: $EXE $DATA_CSV $CHUNK_LEN $OVERLAP $K"
srun "$EXE" "$DATA_CSV" "$CHUNK_LEN" "$OVERLAP" "$K" > "$OUTFILE"
echo "[$(date)] done."
