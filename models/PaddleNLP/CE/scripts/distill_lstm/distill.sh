#unset http_proxy
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/model_compression/distill_lstm/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/models_repo

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

DEVICE=$1

if [[ $3 == 'chnsenticorp' ]];then #GPU/cpu
    python bert_distill.py \
        --task_name chnsenticorp \
        --vocab_size 1256608 \
        --max_epoch 1 \
        --lr 1.0 \
        --dropout_prob 0.1 \
        --batch_size 64 \
        --model_name bert-wwm-ext-chinese \
        --teacher_dir small_models/chnsenticorp/step_100.pdparams \
        --vocab_path senta_word_dict.txt \
        --output_dir distilled_models/chnsenticorp \
        --save_steps 100 \
        --device ${DEVICE} >$log_path/distill_$3_$2_${DEVICE}.log 2>&1
    print_info $? distill_$3_$2_${DEVICE}
elif [[ $3 == 'sst-2' ]];then #GPU
    python bert_distill.py \
        --task_name sst-2 \
        --vocab_size 30522 \
        --max_epoch 1 \
        --lr 1.0 \
        --task_name sst-2 \
        --dropout_prob 0.2 \
        --batch_size 128 \
        --model_name bert-base-uncased \
        --output_dir distilled_models/SST-2 \
        --teacher_dir small_models/SST-2/step_1000.pdparams \
        --save_steps 1000 \
        --device ${DEVICE} \
        --embedding_name w2v.google_news.target.word-word.dim300.en >$log_path/distill_$3_$2_${DEVICE}.log 2>&1
    print_info $? distill_$3_$2_${DEVICE}
else
    python bert_distill.py \
        --task_name qqp \
        --vocab_size 30522 \
        --max_epoch 1 \
        --lr 1.0 \
        --dropout_prob 0.2 \
        --batch_size 256 \
        --model_name bert-base-uncased \
        --n_iter 10 \
        --output_dir distilled_models/QQP \
        --teacher_dir small_models/QQP/step_1000.pdparams \
        --save_steps 1000 \
        --device ${DEVICE} \
        --embedding_name w2v.google_news.target.word-word.dim300.en >$log_path/distill_$3_$2_${DEVICE}.log 2>&1
    print_info $? distill_$3_$2_${DEVICE}
fi
#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
