#!/usr/bin/env bash
# to run:
#   chmod +x ./bin/method/classification/random_baseline.sh
#   ./bin/method/classification/random_baseline.sh
#   nohup ./bin/method/classification/random_baseline.sh > ./logs/method/random_baseline_nohup.log 2>&1 &

set -euo pipefail

datasets=("MONSTER")
subsets=(
  "AudioMNIST"
  "AudioMNIST-DS"
  "CornellWhaleChallenge"
  "FordChallenge"
  "FruitFlies"
  "InsectSound"
  "Pedestrian"
  "Traffic"
)
n_yield_rows=1000
modes=("uniform" "prior" "majority")
seed=42

for dataset in "${datasets[@]}"; do
  for subset in "${subsets[@]}"; do
    for mode in "${modes[@]}"; do
      input_path="./data/datasets/classification/${dataset}/${subset}"
      output_path="./data/generations/classification/${dataset}/${subset}/${mode}.jsonl"

      echo "------------------------------------------------------------"
      echo "Running random | dataset=${dataset} | subset=${subset} | mode=${mode}"
      echo "Input : ${input_path}"
      echo "Output: ${output_path}"
      echo "n_yield_rows: ${n_yield_rows}"

      mkdir -p "$(dirname "${output_path}")"

      python ./src/method/classification/random_baseline.py \
        --input_path "${input_path}" \
        --output_path "${output_path}" \
        --n_yield_rows "${n_yield_rows}" \
        --mode "${mode}" \
        --seed $seed
    done
  done
done
