#!/usr/bin/env bash
#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'cpu' cpu 训练
#获取当前路径
cur_path=`pwd`
model_name=$2
temp_path=$(echo $2|awk -F '_' '{print $2}')

echo "$2 infer"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleRec/models/rank/${temp_path}
log_path=$root_path/log/rank/
mkdir -p $log_path
#临时环境更改

#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
#    cat ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/F_$2.log
    echo -e "\033[31m ${log_path}/F_$2 \033[0m"
else
    cat ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/S_$2.log
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
fi
}

cd $code_path
echo -e "\033[32m `pwd` infer \033[0m";

sed -i "s/  epochs: 4/  epochs: 1/g" config_bigdata.yaml
sed -i "s/  infer_end_epoch: 4/  infer_end_epoch: 1/g" config_bigdata.yaml


# rec python_infer
nohup python -u ../../../tools/paddle_infer.py --model_file=output_model_all_wide_deep/3/rec_inference.pdmodel \
--params_file=output_model_all_wide_deep/3/rec_inference.pdiparams \
--use_gpu=False --data_dir=../../../datasets/criteo/slot_test_data_full \
--reader_file=criteo_reader.py --batchsize=5 > wide_python_infer_e3 &

# 全量数据的infer时间比较久
nohup python -u ../../../tools/paddle_infer.py --model_file=output_model_all_wide_deep/3/rec_inference.pdmodel \
--params_file=output_model_all_wide_deep/2/rec_inference.pdiparams \
--use_gpu=False --data_dir=data/sample_data/train \
--reader_file=criteo_reader.py --batchsize=5 > wide_python_infer_demo &


