#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/simultaneous_translation/$model_name

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

EPOCHS=$5
MAX_STEPS=$6
SAVE_STEPS=$7
LOGGING_STEPS=$8
#访问RD程序
cd $code_path
# 覆盖原来的参数
if [[ $4 != 'con' ]];then
  # 天级别任务覆盖原来的参数，收敛性任务保留
  sed -i "s/save_step: 10000/save_step: ${SAVE_STEPS}/g" config/transformer.yaml
  sed -i "s/print_step: 100/print_step: ${LOGGING_STEPS}/g" config/transformer.yaml
  sed -i "s/max_iter: None/max_iter: ${MAX_STEPS}/g" config/transformer.yaml
  sed -i "s/epoch: 30/epoch: ${EPOCHS}/g" config/transformer.yaml
  sed -i "s/batch_size: 4096/batch_size: 500/g" config/transformer.yaml
  sed -i "s/init_from_params: \"trained_models\/step_final\/\"/init_from_params: \"trained_models\/step_${SAVE_STEPS}\/\"/g" config/transformer.yaml
fi

python -m paddle.distributed.launch --gpus "$3" train.py \
  --config ./config/transformer.yaml
