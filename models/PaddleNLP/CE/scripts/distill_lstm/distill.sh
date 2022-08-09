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
    python bert_distill.py \
        --task_name chnsenticorp \
        --vocab_size 1256608 \
        --max_epoch 1 \
        --lr 1.0 \
        --dropout_prob 0.1 \
        --batch_size 64 \
        --model_name bert-wwm-ext-chinese \
        --teacher_dir ./chnsenticorp/best_bert_wwm_ext_model_880 \
        --vocab_path senta_word_dict.txt \
        --output_dir distilled_models/chnsenticorp \
        --save_steps 100 \
        --device ${DEVICE}
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
        --teacher_dir ./SST-2/best_model_610 \
        --save_steps 1000 \
        --device ${DEVICE} \
        --embedding_name w2v.google_news.target.word-word.dim300.en
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
        --teacher_dir ./QQP/best_model_17000 \
        --save_steps 1000 \
        --device ${DEVICE} \
        --embedding_name w2v.google_news.target.word-word.dim300.en
fi
