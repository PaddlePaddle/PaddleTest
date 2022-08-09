#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/text_classification/pretrained_models/

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

#访问RD程序
cd $code_path


if [[ $1 == 'gpu' ]];then #GPU
    python -m paddle.distributed.launch --gpus "$3" train.py\
        --batch_size 8\
        --epochs 1 \
        --save_dir ./checkpoints\
        --use_amp False \
        --device $1
else #CPU
    python -m train.py\
        --batch_size 8\
        --epochs 1 \
        --save_dir ./checkpoints\
        --use_amp False \
        --device $1

fi
