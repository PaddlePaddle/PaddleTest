# 动态图模型推理
# work_path: PaddleLLM/llm
model_name=$1
ngpus=${2:-8}
step_name=${3:-"grpo"}
save_steps=${4:-1200}

# 1.设置模型路径
if [ "$step_name" == "ppo" ] || [ "$step_name" == "grpo" ] || [ "$step_name" == "reinforce_plus_plus" ]; then 
    model_name_or_path=./checkpoints/$model_name/${step_name}/policy/checkpoint-${save_steps}
else
    model_name_or_path=./checkpoints/$model_name/${step_name}
fi

# 2.设置GPU
if [ $ngpus -eq 0 ]; then  
    DEVICE="0"  
elif [ $ngpus -eq 1 ]; then  
    DEVICE="1"  
elif [ $ngpus -eq 2 ]; then  
    DEVICE="2,3"  
elif [ $ngpus -eq 4 ]; then  
    DEVICE="4,5,6,7"  
elif [ $ngpus -eq 8 ]; then  
    DEVICE="0,1,2,3,4,5,6,7" 
else  
    echo "Unsupported number of GPUs"  
    exit 1  
fi  
export CUDA_VISIBLE_DEVICES=${DEVICE}
current_path=$(pwd)
repo_path=${current_path%%PaddleLLM*}PaddleLLM
llm_path=${repo_path}/llm
export PYTHONPATH=$repo_path:$PYTHONPATH
export PYTHONPATH=$llm_path:$PYTHONPATH


python ./predict/predictor.py \
    --model_name_or_path $model_name_or_path \
    --inference_model \
    --dtype float16