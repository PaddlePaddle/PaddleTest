
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#访问RD程序
cd $code_path

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

if [[ $4 == 'con' ]];then
  # 收敛：
  MAXSTEP=1000000
  SAVE_STEP=20000
  EPOCHS=3
  LOGSTEP=100
  wget https://bert-data.bj.bcebos.com/benchmark_sample%2Fhdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar.gz
  tar -xzvf benchmark_sample%2Fhdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar.gz
  DATATPATH='./hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/training/'
else
  MAXSTEP=3
  SAVE_STEP=1
  EPOCHS=1
  LOGSTEP=1
  DATATPATH='./data/'
fi

echo $MAXSTEP
echo $SAVE_STEP
echo $EPOCHS

if [[ $1 == 'gpu' ]];then #GPU
    python -m paddle.distributed.launch --gpus "$3"  run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs ${EPOCHS} \
    --input_dir ${DATATPATH} \
    --output_dir pretrained_models_multi/ \
    --logging_steps ${LOGSTEP} \
    --save_steps ${SAVE_STEP} \
    --max_steps ${MAXSTEP} \
    --device $1 \
    --use_amp False > $log_path/train_$2_$1.log 2>&1
    print_info $? train_$2_$1

else # cpu
    python run_pretrain.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --max_predictions_per_seq 20 \
        --batch_size 32   \
        --learning_rate 1e-4 \
        --weight_decay 1e-2 \
        --adam_epsilon 1e-6 \
        --warmup_steps 10000 \
        --num_train_epochs 1 \
        --input_dir data/ \
        --output_dir pretrained_models/ \
        --logging_steps 1 \
        --save_steps 1 \
        --max_steps 3 \
        --device $1 \
        --use_amp False > $log_path/train_$2_$1.log 2>&1
    print_info $? train_$2_$1
fi
