#!/usr/bin/env bash
# Run every (experiment, seed) job in experiments.json, PARALLEL jobs at a time.
# Finished jobs (runs/<exp>/seed<k>/DONE) are skipped, so the script can be re-run after an interruption.
#
#   PARALLEL=8 THREADS=4 bash run_all.sh            # all experiments
#   PARALLEL=8 THREADS=4 bash run_all.sh ref_mila    # selected experiments only
set -euo pipefail
cd "$(dirname "$0")"
PARALLEL="${PARALLEL:-8}"
THREADS="${THREADS:-4}"
OUT_ROOT="${OUT_ROOT:-runs}"
mkdir -p logs "$OUT_ROOT"

python - "$@" <<'EOF' > logs/jobs.txt
import json, sys
plan = json.load(open("experiments.json"))
wanted = sys.argv[1:] or list(plan["experiments"])
for name in wanted:
    cfg = plan["experiments"][name]
    for s in plan["seeds_main" if cfg["seeds"] == "main" else "seeds_ablation"]:
        print(name, s)
EOF
echo "$(wc -l < logs/jobs.txt) jobs, PARALLEL=$PARALLEL, THREADS=$THREADS, OUT_ROOT=$OUT_ROOT"

xargs -P "$PARALLEL" -L 1 bash -c \
  'python run_experiment.py --exp "$0" --seed "$1" --threads '"$THREADS"' --out-root '"$OUT_ROOT"' \
     > "logs/$0_seed$1.log" 2>&1 && echo "done $0 seed$1" || echo "FAILED $0 seed$1 (see logs/$0_seed$1.log)"' \
  < logs/jobs.txt | tee -a logs/progress.txt
echo "all jobs finished: $(date)"
