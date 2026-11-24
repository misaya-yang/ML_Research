#!/bin/zsh
# One-command reproduction for the first-answer steering experiment.
#   reproduce.sh dryrun          tests + frozen check + scripted-model dry run of every stage (no weights)
#   reproduce.sh realdry         real model, all arms, 2 development items          (APPROVE=1)
#   reproduce.sh stage0          same-prefix fork replay, 16 development items      (APPROVE=1)
#   reproduce.sh stage1          sampler tuning on development items                 (APPROVE=1)
#   reproduce.sh stage2          pilot, 40 fresh MBPP+ tasks, base + reworded        (APPROVE=1)
#   reproduce.sh stage3          transfer, 40 HumanEval tasks                        (APPROVE=1)
# Real-model stages wait for a free slot: at most one other local model job and
# at least 10 GB of reclaimable memory (keeps >= 6 GB free after loading).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/aidemo/bin/python}
FA=experiments/first_answer
C=$FA/configs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
       HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
stage=${1:-dryrun}
rd() { python3 -c "import json;print(json.load(open('$1'))['run_dir'])"; }
others() { ps aux | grep -E "envs/.*/bin/python" | grep -v grep | grep -v -E "resource_tracker|ipykernel|jupyter|snapshot_download|pytest" | wc -l | tr -d ' '; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+p)*ps/1e9}'; }
wait_slot() {
  while [ "$(others)" -gt 1 ] || [ "$(avail_gb)" -lt 10 ]; do
    echo "$(date '+%F %T') waiting: $(others) other model jobs, $(avail_gb) GB reclaimable"; sleep 60; done
}
need_approve() { [ "${APPROVE:-0}" = "1" ] || { echo "refusing real-model stage '$stage' without APPROVE=1"; exit 2; }; }

case $stage in
  dryrun)
    D=${DRY:-$(mktemp -d)}
    nice -n 19 $PY -m pytest -q $FA/tests -p no:cacheprovider
    $PY $FA/verify_frozen.py
    nice -n 19 $PY $FA/fork_replay.py --config $C/stage0_fork_replay.json --backend fake --grader docker --limit 4 --run-dir $D/stage0
    nice -n 19 $PY $FA/run_arms.py --config $C/dryrun_dev.json --backend fake --grader docker --limit 3 --run-dir $D/arms
    $PY $FA/analyze_arms.py $D/arms --config $C/dryrun_dev.json
    echo "dry-run outputs in $D" ;;
  realdry)
    need_approve; $PY $FA/verify_frozen.py; wait_slot
    D=${DRY:?set DRY to an output directory}
    nice -n 19 $PY $FA/run_arms.py --config $C/dryrun_dev.json --backend hf --approve --limit 2 --run-dir $D
    $PY $FA/analyze_arms.py $D --config $C/dryrun_dev.json ;;
  stage0)
    need_approve; $PY $FA/verify_frozen.py; wait_slot
    nice -n 19 $PY $FA/fork_replay.py --config $C/stage0_fork_replay.json --backend hf --approve ;;
  stage1)
    need_approve; $PY $FA/verify_frozen.py; wait_slot
    nice -n 19 $PY $FA/run_arms.py --config $C/stage1_sampler_tuning.json --backend hf --approve ;;
  stage2|stage3)
    need_approve; $PY $FA/verify_frozen.py; wait_slot
    cfg=$C/$([ $stage = stage2 ] && echo stage2_pilot || echo stage3_transfer).json
    nice -n 19 $PY $FA/run_arms.py --config $cfg --backend hf --approve
    $PY $FA/analyze_arms.py "$(rd $cfg)" --config $cfg ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
