#!/usr/bin/env bash
# 3218600
# to run:
#   chmod +x ./bin/method/baseline.sh
#   ./bin/method/baseline.sh
#   nohup ./bin/method/baseline.sh > ./logs/method/baseline_nohup.log 2>&1 &
#   tail -f ./logs/method/baseline_nohup.log 2>&1 &


set -euo pipefail

# === Config ===
datasets=(
  MCQ1
  TimerBed
  # TSQA
)


prompts=(
  baseline
)

shots=(
  few_shot
  zero_shot
)

# Pick your models here. Llama requires gated access from Meta.
model_types=(
  gpt
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
  test
)

batch_size=16
device_map=(0)         # single GPU id; baseline.py expects a list, but one value is fine
temperature=0.7
show_prompt=0         # 1 to echo prompts, 0 to hide

# === Loop ===
for "$dataset" in "${datasets[@]}"; do

  if [[$dataset == "TimerBed"]]; then
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
        for prompt in "${prompts[@]}"; do
          for shot in "${shots[@]}"; do

            # Your dataset writer saves to: ./data/datasets/<DATASET>/<SPLIT>.jsonl
            input_path="./data/datasets/${dataset}/${subset}/${split}.jsonl"
            output_path="./data/generations/${prompt}/${model_type}/${dataset}/${subset}/${shots}/${split}.jsonl"
            prompt_path="./src/utils/prompts/${prompt}/${shot}.yaml"

            echo "------------------------------------------------------------"
            echo "Running split=${split} | dataset=${dataset} | model=${model_type}"
            echo "Input : ${input_path}"
            echo "Output: ${output_path}"
            echo "Args  : batch_size=${batch_size}, device_map=${device_map}, temp=${temperature}, show_prompt=${show_prompt}"

            # Ensure output directory exists
            mkdir -p "$(dirname "${output_path}")"

            python ./src/baseline.py \
              --input_path "${input_path}" \
              --prompt_path "${prompt_path}" \
              --output_path "${output_path}" \
              --model_type "${model_type}" \
              --batch_size "${batch_size}" \
              --temperature "${temperature}" \
              --device_map "${device_map}" \
              --show_prompt "${show_prompt}"
          done
        done
      done
    done
  done
done
