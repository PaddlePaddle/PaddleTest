
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型训练阶段"

#配置目标数据存储路径
code_path=${nlp_dir}/examples/text_to_knowledge/nptag


# 准备数据
cd $code_path
python -m paddle.distributed.launch --gpus "$3" train.py \
    --batch_size 64 \
    --learning_rate 1e-6 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir ./output/$2 \
    --device $1
