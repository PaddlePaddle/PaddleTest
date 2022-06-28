#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型样例测试阶段"

#路径配置
code_path=${nlp_dir}/examples/machine_translation/seq2seq/

#访问RD程序
cd $code_path

DEVICE=$1


python predict.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --init_from_ckpt attention_models/final \
    --infer_output_file infer_output.txt \
    --beam_size 10 \
    --device ${DEVICE}
