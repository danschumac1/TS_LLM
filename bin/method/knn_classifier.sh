#!/usr/bin/env bash
# kNN-DTW baseline generator (non-LLM)
# to run:
#   chmod +x ./bin/method/knn_time.sh
#   ./bin/method/knn_time.sh
#   nohup ./bin/method/knn_time.sh > ./logs/method/knn_time_nohup.log 2>&1 &

set -euo pipefail

datasets=("TimerBed")
subsets=("CTU" "ECG" "EMG" "HAR" "TEE")
splits=("test")                 # you can add "dev" etc.

# ---- Hyperparams (expand as you like) ----
ks=(1 3)
metrics=("dtw" "euclidean")
sakoe_radii=(10)                # only used for metric=dtw
downs=(2)                       # temporal stride downsampling
n_jobs=-1

gen_root="./data/generations/knn_classifier"
logs_dir="./logs"
mkdir -p "${logs_dir}"

for dataset in "${datasets[@]}"; do
  for subset in "${subsets[@]}"; do
    # auto-derive train set
    train_path="./data/datasets/${dataset}/${subset}/train.jsonl"

    for split in "${splits[@]}"; do
      input_path="./data/datasets/${dataset}/${subset}/${split}.jsonl"

      for k in "${ks[@]}"; do
        for metric in "${metrics[@]}"; do
          if [[ "${metric}" == "dtw" ]]; then
            for radius in "${sakoe_radii[@]}"; do
              for ds in "${downs[@]}"; do
                param_slug="k=${k}_metric=${metric}_band=${radius}_ds=${ds}"
                out_dir="${gen_root}/${dataset}/${subset}/${param_slug}"
                out_path="${out_dir}/${split}.jsonl"
                mkdir -p "${out_dir}"

                echo "------------------------------------------------------------"
                echo "GEN  dataset=${dataset} subset=${subset} split=${split}"
                echo "     ${param_slug}"
                echo "Input : ${input_path}"
                echo "Output: ${out_path}"

                python ./src/knn_time.py \
                  --input_path "${input_path}" \
                  --train_path "${train_path}" \
                  --output_path "${out_path}" \
                  --k "${k}" \
                  --metric "${metric}" \
                  --sakoe_radius "${radius}" \
                  --downsample "${ds}" \
                  --n_jobs "${n_jobs}"
              done
            done
          else
            # euclidean ignores sakoe radius
            for ds in "${downs[@]}"; do
              param_slug="k=${k}_metric=${metric}_ds=${ds}"
              out_dir="${gen_root}/${dataset}/${subset}/${param_slug}"
              out_path="${out_dir}/${split}.jsonl"
              mkdir -p "${out_dir}"

              echo "------------------------------------------------------------"
              echo "GEN  dataset=${dataset} subset=${subset} split=${split}"
              echo "     ${param_slug}"
              echo "Input : ${input_path}"
              echo "Output: ${out_path}"

              python ./src/knn_time.py \
                --input_path "${input_path}" \
                --train_path "${train_path}" \
                --output_path "${out_path}" \
                --k "${k}" \
                --metric "${metric}" \
                --downsample "${ds}" \
                --n_jobs "${n_jobs}"
            done
          fi
        done
      done
    done
  done
done
