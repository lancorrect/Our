

for a in  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  for b in  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
  do
    for g in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
      MODEL_NAME=gcn \
      DATASET=laptop \
      SEED=1000 \
      NUM_EPOCH=50 \
      VOCAB_DIR=./dataset/Laptops_corenlp \
      FUSION=True \
      ALPHA=$a \
      BETA=$b \
      GAMA=$g \
      bash run.sh
    done
  done
done

python ./find_best_result.py
