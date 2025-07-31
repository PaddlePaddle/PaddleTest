# 静态图模型推理
# work_path: PaddleLLM/llm
model_name=$1
ngpus=${2:-8}
step_name=${3:-"grpo"}
save_steps=${4:-1200}

# 1.动转静模型路径
if [ "$step_name" == "ppo" ] || [ "$step_name" == "grpo" ] || [ "$step_name" == "reinforce_plus_plus" ]; then 
    model_name_or_path=./checkpoints/$model_name/${step_name}/policy/checkpoint-${save_steps}
else
    model_name_or_path=./checkpoints/$model_name/${step_name}
fi
output_path="$model_name_or_path/inference"

current_path=$(pwd)
repo_path=${current_path%%PaddleLLM*}PaddleLLM
llm_path=${repo_path}/llm
export PYTHONPATH=$repo_path:$PYTHONPATH
export PYTHONPATH=$llm_path:$PYTHONPATH

# 2. 静态图导出
echo "静态图导出..."
python ./predict/export_model.py \
    --model_name_or_path ${model_name_or_path} \
    --inference_model \
    --output_path ${output_path} \
    --dtype float16

# 3. 静态图推理
echo "静态图推理..."
python ./predict/predictor.py \
    --model_name_or_path ${output_path} \
    --inference_model \
    --dtype "float16" \
    --mode "static"
