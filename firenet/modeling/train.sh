#!/bin/bash

# Define the script parameters
CONFIG="../cfgs/convlstm/full_run.yaml"
TRAINER="../cfgs/trainer_single_gpu.yaml"
DATA="../cfgs/data_monotemporal_full_features.yaml"
SEED=0
MAX_EPOCHS=200
DATA_DIR="../data/WildfireSpreadTS/raw"

# Run the training script
python3 train.py \
    --config=$CONFIG \
    --trainer=$TRAINER \
    --data=$DATA \
    --seed_everything=$SEED \
    --trainer.max_epochs=$MAX_EPOCHS \
    --do_test=True \
    --data.data_dir=$DATA_DIR