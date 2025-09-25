#!/usr/bin/env bash
# to run:
#   chmod +x ./bin/eval/llm_eval.sh
#   ./bin/eval/llm_eval.sh
#   nohup ./bin/eval/llm_eval.sh > ./logs/eval/llm_eval_nohup.log 2>&1 &
#   tail -f ./logs/eval/llm_eval_nohup.log

set -euo pipefail

# === Config (edit me) ===
datasets=(
  # MCQ1
  TimerBed
  # TSQA
)

methods=( # method really
  baseline
  vl_time
)

shots=(
  fs
  zs
)

model_types=(
  gpt
  # llama
  # mistral
  # gemma
)

splits=(
  test
  # train
  # dev
)

# Where generations live; eval reads these
gen_root="./data/generations"

# Optional: where to store tiny eval summaries (stdout is still printed)
eval_root="./data/results"

mkdir -p "${eval_root}"

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
        for method in "${methods[@]}"; do
          for shot in "${shots[@]}"; do
            input_path="${gen_root}/${method}/${model_type}/${dataset}/${subset}/${shot}/${split}.jsonl"
            stamp="$(date +%Y%m%d_%H%M%S)"

            echo "------------------------------------------------------------"
            echo "EVAL  dataset=${dataset} | model=${model_type} | method=${method} | shot=${shot} | split=${split}"
            echo "Input : ${input_path}"

            if [[ ! -f "${input_path}" ]]; then
              echo "SKIP: missing file ${input_path}"
              echo "SKIP: ${input_path}" >> "${summary_path}"
              continue
            fi

            # Run eval; print to stdout and also save a small summary file
            python ./src/eval/eval.py --input_path "${input_path}" 

          done
        done
      done
    done
  done
done