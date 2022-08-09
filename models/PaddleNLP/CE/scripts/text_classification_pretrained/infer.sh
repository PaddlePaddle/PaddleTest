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
    python predict.py\
        --device $1\
        --params_path checkpoints/model_900/model_state.pdparams

else #CPU 跟GPU只是log后缀不一样，也可以传两个参数保持跟GPU一直，这里就可以省掉
    python predict.py\
        --device $1\
        --params_path checkpoints/model_900/model_state.pdparams

fi
