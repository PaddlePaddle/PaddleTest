cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型Fine-tune阶段"

#路径配置
code_path=${nlp_dir}/examples/language_model/$model_name/


#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

MAX_STEPS=$3
SAVE_STEPS=$4
LOGGING_STEPS=$5

cd $code_path
if [ $1 == 'CoLA' ];then
    MAX_STEPS=100
fi
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "$2" ./run_glue.py \
    --model_name_or_path xlnet-base-cased \
    --task_name $1 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_steps ${MAX_STEPS}\
    --output_dir ./$1/
