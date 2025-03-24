# 动态图模型推理
model_name=$1
ngpus=${2:-8}
step_name=${3:-"grpo"}
# 1.设置模型路径
if [ "$step_name" == "ppo" ] || [ "$step_name" == "grpo" ]; then 
    model_name_or_path=./checkpoints/$model_name/${step_name}/policy/checkpoint-20
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


python ./predict/predictor.py \
    --model_name_or_path $model_name_or_path \
    --inference_model \
    --dtype float16