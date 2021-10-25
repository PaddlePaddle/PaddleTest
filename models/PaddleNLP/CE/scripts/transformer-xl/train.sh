#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

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
    python -m paddle.distributed.launch --gpus $2 train.py --config ./configs/enwik8.yaml > $log_path/multi_enwik8_train.log 2>&1
    print_info $? "multi_enwik8_train"
elif [ $1 == 'recv' ]; then
    #恢复训练
    sed -i "s#init_from_checkpoint: .*#init_from_checkpoint: \"./trained_models/step_5\"#g" $code_path/configs/enwik8.yaml
    sed -i "s#init_from_params: .*#init_from_params: \"./trained_models/step_5\"#g" $code_path/configs/enwik8.yaml
    python train.py --config ./configs/enwik8.yaml > $log_path/recv_enwik8_train.log 2>&1
    print_info $? "recv_enwik8_train"
else #单卡
    sed -i "s#init_from_checkpoint: .*#init_from_checkpoint: \"\"#g" $code_path/configs/enwik8.yaml
    sed -i "s#init_from_params: .*#init_from_params: \"\"#g" $code_path/configs/enwik8.yaml
    python train.py --config ./configs/enwik8.yaml > $log_path/single_enwik8_train.log 2>&1
    print_info $? "single_enwik8_train"
fi
