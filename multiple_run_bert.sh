#!/bin/bash
# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0

for a in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
do
    for b in 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for g in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        do
            MODEL_NAME=gcnbert \
            DATASET=restaurant \
            SEED=1000 \
            NUM_EPOCH=15 \
            FUSION=True \
            ALPHA=$a \
            BETA=$b \
            GAMA=$g \
            bash run.sh
        done
    done
done

python ./find_best_result.py