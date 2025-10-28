#!/bin/bash
# to run:
#   chmod +x ./bin/method/classification/InstructionTime/train_universal.sh
#   ./bin/method/classification/InstructionTime/train_universal.sh


# Run to train Instructtime-Universal
python main.py \
--save_path $VQVAE_PATH \
--dataset $DATASET \
--model_path $DATA_PATH \
--device $DEVICE \
--adapt False
