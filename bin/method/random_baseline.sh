#!/usr/bin/env bash
# 945026
# to run:
#   chmod +x ./bin/method/random_baseline.sh
#   ./bin/method/random_baseline.sh
#   nohup ./bin/method/random_baseline.sh > ./logs/method/random_baseline_nohup.log 2>&1 &
#   tail -f ./logs/method/random_baseline_nohup.log

# set -euo pipefail

# === Config ===
datasets=(
  "TimerBed"
)

splits=(
  test
)


# === Loop ===
for dataset in "${datasets[@]}"; do
  if [[ $dataset == "TimerBed" ]]; then
    subsets=(
      # "_TINYTEST"
      "CTU"
      "ECG"
      "EMG"
      "HAR"
      "TEE"
    )
  else
    subsets=("subset")
  fi

  for subset in "${subsets[@]}"; do
    for split in "${splits[@]}"; do

      input_path="./data/datasets/${dataset}/${subset}/${split}.jsonl"
      output_path="./data/generations/random/${dataset}/${subset}/${split}.jsonl"

      echo "------------------------------------------------------------"
      echo "Running RANDOM baseline"
      echo "Split   : ${split}"
      echo "Dataset : ${dataset}"
      echo "Subset  : ${subset}"
      echo "Input   : ${input_path}"
      echo "Output  : ${output_path}"
      echo "Args    : batch_size=${batch_size}, seed=${seed}"

      # Ensure output directory exists
      mkdir -p "$(dirname "${output_path}")"

      python ./src/method/random_benchmark.py \
        --input_path "${input_path}" \
        --output_path "${output_path}" 
    done
  done
done
