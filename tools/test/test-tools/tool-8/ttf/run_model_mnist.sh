python mnist_main.py \
  --model_dir=./mnist_model \
  --data_dir=./mnist_data \
  --distribution_strategy=one_device \
  --num_gpus=1 \
  --download >log.tmp 2>&1

now=`date +'%Y-%m-%d %H:%M:%S'`
start_time=$(date --date="$now" +%s)
python mnist_main.py \
  --model_dir=./mnist_model \
  --data_dir=./mnist_data \
  --train_epochs=1 \
  --distribution_strategy=one_device \
  --num_gpus=1 >log.tmp 2>&1
now=`date +'%Y-%m-%d %H:%M:%S'`
end_time=$(date --date="$now" +%s)

used_time=$((end_time-start_time))
ips=`awk 'BEGIN{printf "%.2f\n",60000/'$used_time'}'`
echo -n {'"ips:"' $ips}
