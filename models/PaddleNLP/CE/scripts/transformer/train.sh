#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/machine_translation/$model_name
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

#访问RD程序
cd $code_path

if [[ $4 != 'con' ]];then
    # 非收敛任务,修改yaml中的参数
    sed -i "s/save_step: 10000/save_step: 10/g" configs/transformer.base.yaml
    sed -i "s/print_step: 100/print_step: 10/g" configs/transformer.base.yaml
    sed -i "s/epoch: 30/epoch: 1/g" configs/transformer.base.yaml
    sed -i "s/max_iter: None/max_iter: 30/g" configs/transformer.base.yaml
    sed -i 's/init_from_params: \".\/trained_models\/step_final\/\"/init_from_params: \".\/trained_models\/step_30\/\"/g' configs/transformer.base.yaml

fi

#判断CPU还是GPU
if [ $2 == "cpu" ];then
    sed -i "s/use_gpu: True/use_gpu: False/g" configs/transformer.base.yaml
fi

#训练
if [ $1 == "multi" ];then #多卡
    python -m paddle.distributed.launch\
        --gpus $3 train.py\
        --config ./configs/transformer.base.yaml > $log_path/multi_cards_train.log 2>&1
    cat $log_path/multi_cards_train.log
    cat $code_path/log/workerlog.0

else #单卡或CPU
    python train.py --config ./configs/transformer.base.yaml > $log_path/single_card_train.log 2>&1
    cat $log_path/single_card_train.log
fi

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
