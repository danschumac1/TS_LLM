#!/usr/bin/env bash
# Evaluate all kNN generations under a dataset/subset, per split.
# to run:
#   chmod +x ./bin/eval/knn_eval.sh
#   ./bin/eval/knn_eval.sh
#   nohup ./bin/eval/knn_eval.sh > ./logs/eval/knn_eval_nohup.log 2>&1 &

set -euo pipefail

datasets=(
    "TimerBed"
    )

subsets=(
    "CTU" 
    # "ECG" 
    "EMG" 
    "HAR" 
    "TEE"
    )

splits=(
    "test"
    )

gen_root="./data/generations/knn_classifier"
results_root="./data/results/knn_classifier"
mkdir -p "${results_root}"

for dataset in "${datasets[@]}"; do
  for subset in "${subsets[@]}"; do
    base_dir="${gen_root}/${dataset}/${subset}"
    [[ -d "${base_dir}" ]] || { echo "SKIP: ${base_dir} not found"; continue; }

    # each subdir is a param slug, e.g., k=1_metric=dtw_band=10_ds=2
    for param_dir in "${base_dir}"/*; do
      [[ -d "${param_dir}" ]] || continue
      param_slug="$(basename "${param_dir}")"

      for split in "${splits[@]}"; do
        input_path="${param_dir}/${split}.jsonl"
        if [[ ! -f "${input_path}" ]]; then
          echo "SKIP: missing ${input_path}"
          continue
        fi

        stamp="$(date +%Y%m%d_%H%M%S)"

        echo "------------------------------------------------------------"
        echo "EVAL dataset=${dataset} subset=${subset} split=${split}"
        echo "     params=${param_slug}"
        echo "Input : ${input_path}"

        # Reuse your existing src/eval.py; it expects one input_path
        python ./src/eval/eval.py --input_path "${input_path}" 
      done
    done
  done
done
