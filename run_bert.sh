#!/bin/bash

python ./train.py --model_name $MODEL_NAME --dataset $DATASET --seed $SEED  --bert_lr 2e-5 --num_epoch $NUM_EPOCH --hidden_dim 768 --max_length 100 --cuda 1 --fusion $FUSION --alpha $ALPHA --beta $BETA --gama $GAMA