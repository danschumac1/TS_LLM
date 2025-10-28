#!/bin/bash
# to run:
#   chmod +x ./bin/method/classification/InstructionTime/train_adapt.sh
#   ./bin/method/classification/InstructionTime/train_adapt.sh


# Run to train Instructtime-Adapt
VQVAE_PATH="data/generations/classification/InstructionTime"
DATASET="ECG"
DATA_PATH="data/datasets/TimerBed/ECG/test.jsonl"
DEVICE="cuda"
LR=.01
ADAPT=True

python main.py \
--save_path $VQVAE_PATH \
--dataset $DATASET \
--model_path $DATA_PATH \
--load_model_path $DATA_PATH \
--device $DEVICE \
--lr $LR \
--adapt $ADAPT