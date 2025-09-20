#!/usr/bin/env bash
# 22341
# to run:
#   chmod +x ./bin/method/vl_time.sh
#   ./bin/method/vl_time.sh
#   nohup ./bin/method/vl_time.sh > ./logs/method/vl_timenohup.log 2>&1 &
#   tail -f logs/method/vl_timenohup.log

set -euo pipefail

debug=0
debug_prints=0
shots=(
  "zs"
  "fs"
  )
# shots="zs"


# === Config ===
datasets=(
#   MCQ1
  "TimerBed"
  # TSQA
)


# Pick your models here. Llama requires gated access from Meta.
model_types=(
  "gpt"
  # llama
  # mistral
  # gemma
  # TimeMQA_qwen
  # TimeMQA_llama
  # TimeMQA_mistral   
)

splits=(
  # train
  # dev
  "test"
)

# === Loop ===
for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "TimerBed" ]]; then
    subsets=(
      "CTU"
      "ECG"
      "EMG"
      "HAR"
      "TEE" 
    )
  else
    subsets=(
      "subset"
      )
  fi
  for subset in "${subsets[@]}"; do 
    for model_type in "${model_types[@]}"; do
      for split in "${splits[@]}"; do
        for shot in "${shots[@]}"; do

            # Your dataset writer saves to: ./data/datasets/<DATASET>/<SPLIT>.jsonl
            input_path="./data/datasets/${dataset}/${subset}/${split}.jsonl"
            output_path="./data/generations/vl_time/${model_type}/${dataset}/${subset}/${shots}/${split}.jsonl"

            echo "------------------------------------------------------------"
            echo "Running split=${split} | dataset=${dataset} | model=${model_type}"
            echo "Input : ${input_path}"
            echo "Output: ${output_path}"

            # Ensure output directory exists
            mkdir -p "$(dirname "${output_path}")"

            python ./src/vl_time.py \
              --input_path "${input_path}" \
              --output_path "${output_path}" \
              --model_type "${model_type}" \
              --shots "${shots}" \
              --debug "${debug}" \
              --debug_prints "${debug_prints}"
        done
      done
    done
  done
done
