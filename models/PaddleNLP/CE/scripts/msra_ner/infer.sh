#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型样例测试阶段"

#路径配置
code_path=${nlp_dir}/examples/information_extraction/msra_ner/

#访问RD程序
cd $code_path

DEVICE=$1
MODEL_STEP=$2


python -u ./predict.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 32 \
    --device ${DEVICE} \
    --init_checkpoint_path tmp/msra_ner/model_${MODEL_STEP}.pdparams
