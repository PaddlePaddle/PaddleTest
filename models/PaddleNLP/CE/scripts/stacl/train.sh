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
code_path=$cur_path/../../models_repo/examples/simultaneous_translation/$model_name
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

#访问RD程序
cd $code_path
# 覆盖原来的参数
if [[ $4 != 'con' ]];then
  # 天级别任务覆盖原来的参数，收敛性任务保留
  sed -i 's/save_step: 10000/save_step: 10/g' config/transformer.yaml
  sed -i "s/print_step: 100/print_step: 10/g" config/transformer.yaml
  sed -i "s/max_iter: None/max_iter: 100/g" config/transformer.yaml
  sed -i 's/epoch: 30/epoch: 1/g' config/transformer.yaml
  sed -i "s/batch_size: 4096/batch_size: 500/g" config/transformer.yaml
  sed -i 's/init_from_params: \"trained_models\/step_final\/\"/init_from_params: \"trained_models\/step_10\/\"/g' config/transformer.yaml
fi

python -m paddle.distributed.launch --gpus "$3" train.py \
  --config ./config/transformer.yaml > $log_path/train_$2_$1.log 2>&1

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
