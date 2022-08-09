python mnist_main.py \
  --model_dir=./mnist_model \
  --data_dir=./mnist_data \
  --distribution_strategy=one_device \
  --num_gpus=1 \
  --download >log.tmp 2>&1

python mnist_main.py \
  --model_dir=./mnist_model \
  --data_dir=./mnist_data \
  --train_epochs=1 \
  --distribution_strategy=one_device \
  --num_gpus=1 >log.tmp 2>&1
