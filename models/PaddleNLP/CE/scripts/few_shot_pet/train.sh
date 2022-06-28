#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/few_shot/pet/


cd $code_path

python -u -m paddle.distributed.launch --gpus $3 \
    pet.py \
	--task_name $4 \
	--device $1 \
    --pattern_id 0 \
	--save_dir "checkpoints/$4/$2" \
	--index 0 \
	--batch_size 16 \
	--learning_rate 1E-4 \
	--epochs 1 \
	--max_seq_length 512 \
	--save_steps 100 \
	--rdrop_coef $5 \
	--language_model "ernie-1.0"
