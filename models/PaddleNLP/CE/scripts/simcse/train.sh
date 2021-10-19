
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型训练阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#配置目标数据存储路径
root_path=$cur_path/../../
modle_path=$cur_path/../../models_repo/
code_path=$cur_path/../../models_repo/examples/text_matching/simcse/
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

#准备数据
cd $code_path
python -u -m paddle.distributed.launch --gpus $3 \
  train.py \
  --save_dir ./$4/$2 \
  --batch_size 64 \
  --learning_rate 5E-5 \
  --epochs 1 \
  --save_steps 100 \
  --eval_steps 100 \
  --max_seq_length 64 \
  --infer_with_fc_pooler \
  --dropout 0.1 \
  --train_set_file "./senteval_cn/$4/train.txt" \
  --test_set_file "./senteval_cn/$4/dev.tsv"
  --device $1 >$log_path/train_$2_$4_$1.log 2>&1
print_info $? ttrain_$2_$4_$1
