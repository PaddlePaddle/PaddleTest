
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/applications/sentiment_analysis/pp_minilm/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

#访问RD程序
cd $code_path

python train.py \
    --base_model_name "ppminilm-6l-768h" \
    --train_path "../data/cls_data/train.txt" \
    --dev_path "../data/cls_data/dev.txt" \
    --label_path "../data/cls_data/label.dict" \
    --num_epochs 5 \
    --batch_size 16 \
    --max_seq_len 256 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --warmup_proportion 0.1 \
    --log_steps 50 \
    --eval_steps 100 \
    --seed 1000 \
    --device $1 \
    --checkpoints "../checkpoints/pp_checkpoints/" > $log_path/ppminilm_train_$2_$1.log 2>&1

print_info $? ppminilm_train_$2_$1
