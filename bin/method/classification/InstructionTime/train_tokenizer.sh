#!/bin/bash
# To run:
#   chmod +x ./bin/method/classification/InstructionTime/train_tokenizer.sh
#   ./bin/method/classification/InstructionTime/train_tokenizer.sh

# Available datasets
DATASETS=("cpu" "emg" "har" "lightning" "whale")

DEVICE="cuda"
LR=0.0005
ADAPT=True
D_MODEL=64
WAVE_LENGTH=3
NUM_TOKEN=2560

for DATASET in "${DATASETS[@]}"; do
    SAVE_PATH="./models/classification/TimeMixer/tokenizer/${DATASET}"
    DATA_PATH="./data/datasets/classification/${DATASET}/"

    echo "🔹 Training tokenizer for dataset: ${DATASET}"
    python ./src/method/classification/InstructTime/TStokenizer/main.py \
        --save_path "$SAVE_PATH" \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --device "$DEVICE" \
        --d_model "$D_MODEL" \
        --wave_length "$WAVE_LENGTH" \
        --n_embed "$NUM_TOKEN"
done
