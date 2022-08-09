cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型finetune阶段"

#路径配置
code_path=${nlp_dir}/model_zoo/tinybert/

MAX_STEPS=$4
SAVE_STEPS=$5
LOGGING_STEPS=$6
#访问RD程序
cd $code_path

cd ../../examples/benchmark/glue/

python -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $3 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_steps ${MAX_STEPS} \
    --output_dir ./tmp/$3/$2\
    --device $1
