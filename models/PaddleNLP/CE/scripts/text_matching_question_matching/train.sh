#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/text_matching/question_matching

MAX_STEPS=$4
SAVE_STEPS=$5
EVAL_STEP=$6
#访问RD程序
cd $code_path

if [[ $2 == "multi" ]]; then
    python -u -m paddle.distributed.launch --gpus $3 train.py \
       --train_set train.txt \
       --dev_set dev.txt \
       --eval_step ${EVAL_STEP} \
       --save_dir ./checkpoints/$2 \
       --train_batch_size 32 \
       --learning_rate 2E-5 \
       --epochs 1 \
       --save_step ${SAVE_STEPS} \
       --max_steps ${MAX_STEPS} \
       --rdrop_coef 0.0 \
       --device $1
else
    python train.py \
       --train_set train.txt \
       --dev_set dev.txt \
       --eval_step ${EVAL_STEP} \
       --save_dir ./checkpoints/$2 \
       --train_batch_size 32 \
       --learning_rate 2E-5 \
       --epochs 1 \
       --save_step ${SAVE_STEPS} \
       --max_steps ${MAX_STEPS} \
       --rdrop_coef 0.0 \
       --device $1
fi
