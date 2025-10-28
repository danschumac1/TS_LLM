#!/usr/bin/env bash
# to run:
#   chmod +x ./bin/method/classification/dtw1nn.sh
#   ./bin/method/classification/dtw1nn.sh
#   nohup ...

set -euo pipefail

datasets=(
  "MONSTER"
  )
subsets=(
  "AudioMNIST"
  "AudioMNIST-DS"
  "CornellWhaleChallenge"
  "FordChallenge"
  "FruitFlies"
  "InsectSound"
  "Pedestrian"
  "Traffic"
  )  # add more as needed
n_yield_rows=10

for dataset in "${datasets[@]}"; do
  for subset in "${subsets[@]}"; do
    input_path="./data/datasets/classification/${dataset}/${subset}"
    output_path="./data/generations/dtw1nn/${dataset}/${subset}.jsonl"

    echo "------------------------------------------------------------"
    echo "Running dtw1nn | dataset=${dataset} | subset=${subset}"
    echo "Input : ${input_path}"
    echo "Output: ${output_path}"
    echo "n_yield_rows: ${n_yield_rows}"

    mkdir -p "$(dirname "${output_path}")"

    python ./src/method/classification/dtw1nn.py \
      --input_path "${input_path}" \
      --output_path "${output_path}" \
      --n_yield_rows "${n_yield_rows}"
  done
done
