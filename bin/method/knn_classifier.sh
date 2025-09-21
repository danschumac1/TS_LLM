#!/usr/bin/env bash
# 3218600
# to run:
#   chmod +x ./bin/method/knn_classifier.sh
#   ./bin/method/knn_classifier.sh
#   nohup ./bin/method/knn_classifier.sh > ./logs/method/knn_classifier_nohup.log 2>&1 &
#   tail -f ./logs/method/knn_classifier_nohup.log

# UNCOMMENT IF YOU WANT TO EXIT ON FAIL
# set -euo pipefail

debug=1

datasets=("TimerBed")
splits=("test")

# Cross-validation knobs (the Python will tune k/metric/band internally)
cv_folds=5
scoring="accuracy"   # or "macro_f1" if your classes are skewed
seed=66
n_jobs=-1

gen_root="./data/generations/knn_classifier"
mkdir -p "${gen_root}"

for dataset in "${datasets[@]}"; do
  # define subsets safely (works even if you later add more datasets)
  if [[ "${dataset}" == "TimerBed" ]]; then
    subsets=("CTU" "ECG" "EMG" "HAR" "TEE")
  else
    subsets=("subset")
  fi

  for subset in "${subsets[@]}"; do
    train_path="./data/datasets/${dataset}/${subset}/train.jsonl"

    for split in "${splits[@]}"; do
      input_path="./data/datasets/${dataset}/${subset}/${split}.jsonl"

      # Skip if inputs are missing (e.g., ECG has no train)
      if [[ ! -f "${train_path}" ]]; then
        echo "SKIP: missing train ${train_path}"
        continue
      fi
      if [[ ! -f "${input_path}" ]]; then
        echo "SKIP: missing input ${input_path}"
        continue
      fi

      # Output slug reflects that CV picked hyperparams internally
      param_slug="auto_cv=${cv_folds}_${scoring}"
      out_dir="${gen_root}/${dataset}/${subset}/${param_slug}"
      out_path="${out_dir}/${split}.jsonl"
      mkdir -p "${out_dir}"

      echo "------------------------------------------------------------"
      echo "GEN dataset=${dataset} subset=${subset} split=${split}"
      echo "    ${param_slug}"
      echo "Input : ${input_path}"
      echo "Train : ${train_path}"
      echo "Output: ${out_path}"

      python ./src/method/knn_classifier.py \
        --input_path "${input_path}" \
        --train_path "${train_path}" \
        --output_path "${out_path}" \
        --cv_folds "${cv_folds}" \
        --scoring "${scoring}" \
        --seed "${seed}" \
        --n_jobs "${n_jobs}" \
        --debug "${debug}"
    done
  done
done
