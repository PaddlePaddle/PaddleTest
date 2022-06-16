cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型样例评估阶段"

#路径配置
code_path=${nlp_dir}/examples/text_to_knowledge/ernie-ctm/

#访问RD程序
cd $code_path

python -m paddle.distributed.launch --gpus $2 predict.py \
    --params_path ./tmp/model_100/model_state.pdparams \
    --batch_size 32 \
    --device $1

