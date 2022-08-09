#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型预测阶段"

#路径配置
code_path=${nlp_dir}/examples/few_shot/pet/

cd $code_path

mkdir -p ./output/$4

python -u -m paddle.distributed.launch --gpus $3 predict.py \
    --task_name $4 \
    --device $1 \
    --init_from_ckpt "./checkpoints/$4/$2/model_$5/model_state.pdparams" \
    --output_dir "./output/$4" \
    --batch_size 32 \
    --max_seq_length 512
