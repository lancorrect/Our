#!/bin/bash

# * laptop

# * GCN
# python ./train.py --model_name $MODEL_NAME --dataset $DATASET --seed $SEED --num_epoch $NUM_EPOCH --vocab_dir $VOCAB_DIR --fusion $FUSION --alpha $ALPHA --beta $BETA --gama $GAMA
# * GCN with Bert
python ./train.py --model_name $MODEL_NAME --dataset $DATASET --seed $SEED  --bert_lr 2e-5 --num_epoch $NUM_EPOCH --hidden_dim 768 --max_length 100 --cuda 0 --fusion $FUSION --alpha $ALPHA --beta $BETA --gama $GAMA


# * restaurant

# * DualGCN
# CUDA_VISIBLE_DEVICES=0 python ./DualGCN/train.py --model_name dualgcn --dataset restaurant --seed 1000 --num_epoch 50 --vocab_dir ./DualGCN/dataset/Restaurants_corenlp --cuda 0 --losstype doubleloss --alpha 0.2 --beta 0.3 --parseadj
# * DualGCN with Bert
# CUDA_VISIBLE_DEVICES=0 python ./DualGCN/train.py --model_name dualgcnbert --dataset restaurant --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 --losstype doubleloss --alpha 0.6 --beta 0.9 --parseadj


# * twitter

# * DualGCN
# CUDA_VISIBLE_DEVICES=0 python ./DualGCN/train.py --model_name dualgcn --dataset twitter --seed 1000 --num_epoch 50 --vocab_dir ./DualGCN/dataset/Tweets_corenlp --cuda 0 --losstype doubleloss --alpha 0.3 --beta 0.2 --parseadj
# * DualGCN with Bert
# CUDA_VISIBLE_DEVICES=0 python ./DualGCN/train.py --model_name dualgcnbert --dataset twitter --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 --losstype doubleloss --alpha 0.5 --beta 0.9 --parseadj

# test
# python ./train.py --model_name gcn --dataset laptop --seed 1000 --num_epoch 50 --vocab_dir ./dataset/Laptops_corenlp --fusion True --alpha 1.0 --beta 0.9 --gama 3.0
# python ./train.py --model_name gcnbert --dataset laptop --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 --fusion True --alpha 1.4 --beta 0.5 --gama 0.1