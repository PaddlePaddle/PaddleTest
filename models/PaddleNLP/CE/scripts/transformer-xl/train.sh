#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/language_model/$model_name/

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

#访问RD程序,替换变量
cd $code_path

sed -i "s/save_step: .*/save_step: 5/g" $code_path/configs/enwik8.yaml
sed -i "s/print_step: .*/print_step: 1/g" $code_path/configs/enwik8.yaml
sed -i "s/epoch: .*/epoch: 1/g" $code_path/configs/enwik8.yaml
sed -i "s/max_step: .*/max_step: 6/g" $code_path/configs/enwik8.yaml
sed -i "s#init_from_checkpoint: .*#init_from_checkpoint: \"\"#g" $code_path/configs/enwik8.yaml
sed -i "s#init_from_params: .*#init_from_params: \"\"#g" $code_path/configs/enwik8.yaml

#
if [ $1 == 'multi' ];then #多卡
    sed -i "s#init_from_checkpoint: .*#init_from_checkpoint: \"\"#g" $code_path/configs/enwik8.yaml
    sed -i "s#init_from_params: .*#init_from_params: \"\"#g" $code_path/configs/enwik8.yaml
    python -m paddle.distributed.launch --gpus $2 train.py --config ./configs/enwik8.yaml
elif [ $1 == 'recv' ]; then
    #恢复训练
    sed -i "s#init_from_checkpoint: .*#init_from_checkpoint: \"./trained_models/step_5\"#g" $code_path/configs/enwik8.yaml
    sed -i "s#init_from_params: .*#init_from_params: \"./trained_models/step_5\"#g" $code_path/configs/enwik8.yaml
    python train.py --config ./configs/enwik8.yaml > $log_path/recv_enwik8_train.log 2>&1
    print_info $? "recv_enwik8_train"
else #单卡
    sed -i "s#init_from_checkpoint: .*#init_from_checkpoint: \"\"#g" $code_path/configs/enwik8.yaml
    sed -i "s#init_from_params: .*#init_from_params: \"\"#g" $code_path/configs/enwik8.yaml
    python train.py --config ./configs/enwik8.yaml
fi
