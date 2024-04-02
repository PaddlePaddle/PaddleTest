#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/model_compression/distill_lstm/

#访问RD程序
cd $code_path

DEVICE=$1

if [[ $3 == 'chnsenticorp' ]];then #GPU/cpu
    python small.py \
        --task_name chnsenticorp \
        --max_epoch 1 \
        --vocab_size 1256608 \
        --batch_size 64 \
        --model_name bert-wwm-ext-chinese \
        --optimizer adam \
        --lr 3e-4 \
        --dropout_prob 0.2 \
        --vocab_path senta_word_dict.txt \
        --device ${DEVICE} \
        --save_steps 100 \
        --output_dir small_models/chnsenticorp/
elif [[ $3 == 'sst-2' ]];then #GPU
    python small.py \
        --task_name $3 \
        --vocab_size 30522 \
        --max_epoch 1 \
        --batch_size 64 \
        --lr 1.0 \
        --dropout_prob 0.4 \
        --output_dir small_models/SST-2 \
        --save_steps 1000 \
        --vocab_path $4 \
        --device ${DEVICE} \
        --embedding_name w2v.google_news.target.word-word.dim300.en
else
    python small.py \
        --task_name qqp \
        --vocab_size 30522 \
        --max_epoch 1 \
        --batch_size 256 \
        --lr 2.0 \
        --dropout_prob 0.4 \
        --output_dir small_models/QQP \
        --save_steps 1000 \
        --vocab_path $4 \
        --device ${DEVICE} \
        --embedding_name w2v.google_news.target.word-word.dim300.en
fi
