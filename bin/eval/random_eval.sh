#!/usr/bin/env bash
# to run:
#   chmod +x ./bin/eval/random_eval.sh
#   ./bin/eval/random_eval.sh
#   nohup ./bin/eval/random_eval.sh > ./logs/eval/random_eval_nohup.log 2>&1 &
#   tail -f ./logs/eval/random_eval_nohup.log

set -euo pipefail

# === Config (edit me) ===
datasets=(
  TimerBed
)

splits=(
  test
  # dev
  # train
)

# Where random generations live
gen_root="./data/generations/random"

# Optional: folder to drop tiny eval summaries (stdout still prints live)
eval_root="./data/results/random"
mkdir -p "${eval_root}"

# === Loop ===
for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "TimerBed" ]]; then
    subsets=(
      "_TINYTEST"
      # "CTU"
      # "ECG"
      # "EMG"
      # "HAR"
      # "TEE"
    )
  else
    subsets=("subset")
  fi

  for subset in "${subsets[@]}"; do
    for split in "${splits[@]}"; do
      input_path="${gen_root}/${dataset}/${subset}/${split}.jsonl"
      stamp="$(date +%Y%m%d_%H%M%S)"
      summary_path="${eval_root}/summary_${dataset}_${subset}_${split}_${stamp}.txt"

      echo "------------------------------------------------------------"
      echo "EVAL  dataset=${dataset} | subset=${subset} | split=${split}"
      echo "Input : ${input_path}"
      echo "Summary: ${summary_path}"

      if [[ ! -f "${input_path}" ]]; then
        echo "SKIP: missing file ${input_path}"
        echo "SKIP: ${input_path}" >> "${summary_path}"
        continue
      fi

      # Run eval; print to stdout and also capture a brief summary
      python ./src/eval/eval.py --input_path "${input_path}" | tee "${summary_path}"

    done
  done
done
