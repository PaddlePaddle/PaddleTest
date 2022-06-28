
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型预测阶段"

#配置目标数据存储路径
code_path=${nlp_dir}/examples/text_to_knowledge/nptag


# 准备数据
cd $code_path
python -m paddle.distributed.launch --gpus $2 predict.py \
    --device=$1 \
    --params_path ./output/single/model_100/model_state.pdparams
