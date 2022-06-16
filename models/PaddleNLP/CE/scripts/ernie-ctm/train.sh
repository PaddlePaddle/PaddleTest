#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/text_to_knowledge/ernie-ctm/

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

#访问RD程序
cd $code_path

python -m paddle.distributed.launch --gpus "$3"  train.py \
    --max_seq_len 128 \
    --batch_size 32   \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir ./tmp/ \
    --device $1

