#!/usr/bin/env bash
# 3218600
# to run:
#   chmod +x ./bin/method/knn_cv.sh
#   ./bin/method/knn_cv.sh
#   nohup ./bin/method/knn_cv.sh > ./logs/method/knn_cv_nohup.log 2>&1 &
#   tail -f ./logs/method/knn_cv_nohup.log

# Fill/append a single TSV with best kNN params per dataset/subset.

# set -euo pipefail
debug=0

datasets=("TimerBed")
cv_folds=5
scoring="accuracy"   # or "macro_f1"
seed=66
n_jobs=-1

params_tsv="./experiments/knn/knn_cv_params.tsv"
mkdir -p "$(dirname "$params_tsv")"

for dataset in "${datasets[@]}"; do
  if [[ "${dataset}" == "TimerBed" ]]; then
    subsets=("CTU" "ECG" "EMG" "HAR" "TEE")
  else
    subsets=("subset")
  fi

  for subset in "${subsets[@]}"; do
    train_path="./data/datasets/${dataset}/${subset}/train.jsonl"
    if [[ ! -f "${train_path}" ]]; then
      echo "SKIP: missing train ${train_path}"
      continue
    fi

    echo "CV: ${dataset}/${subset}"
    python ./src/method/knn_cv.py \
      --train_path "${train_path}" \
      --params_tsv "${params_tsv}" \
      --cv_folds "${cv_folds}" \
      --scoring "${scoring}" \
      --seed "${seed}" \
      --n_jobs "${n_jobs}" \
      --debug "${debug}"
  done
done

echo "Wrote/updated: ${params_tsv}"
