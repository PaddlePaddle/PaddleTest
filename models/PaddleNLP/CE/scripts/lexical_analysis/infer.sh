#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型样例测试阶段"

#路径配置
code_path=${nlp_dir}/examples/lexical_analysis/

#访问RD程序
cd $code_path

DEVICE=$1
MODEL_STEP=$2
python predict.py --data_dir ./lexical_analysis_dataset_tiny \
    --init_checkpoint ./save_dir/model_${MODEL_STEP}.pdparams \
    --batch_size 32 \
    --device ${DEVICE}
