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

convergence_value=`cat log.tmp | grep accuracy_top_1 | grep training_accuracy_top_1 | awk -F '[ |,]' '{print $2}'`
if [ -z ${convergence_value} ];then
        echo -n "trian error"
        exit 1
fi
python is_conv.py ${convergence_value} 0.807 >log.conv 2>&1
flag=$?
return ${flag}
