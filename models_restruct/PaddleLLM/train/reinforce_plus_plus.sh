# work_path: PaddleLLM/llm/alignment/ppo
# reinforce_plus_plus 训练
model_name=$1
ngpus=${2:-8}
steps=${3:-2}
ext_args=""

# 1. 模型准备
echo "清理显存"
# fuser -v /dev/nvidia* 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) print $i}' | xargs kill -9 2>/dev/null
sleep 3s
echo "清理Checkpoints"
rm -rf ../../checkpoints/${model_name}/reinforce_plus_plus/* 2>/dev/null 

if [[ ${model_name} == "qwen" ]]; then
    model_name_or_path="Qwen/Qwen2.5-1.5B"
elif [[ ${model_name} == "llama" ]]; then
    model_name_or_path="meta-llama/Meta-Llama-3-8B"
fi

output_dir="../../checkpoints/${model_name}/reinforce_plus_plus" # 以llm为根目录

# 2. 数据准备 
if [ ! -d "ppo-kk" ]; then
    wget https://paddlenlp.bj.bcebos.com/datasets/examples/ppo-kk.tgz && tar zxf ppo-kk.tgz
fi

# 3. 设置环境变量
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

# 4. 启动训练脚本
echo "启动reward服务"
cd reward
python reward_server.py > reward_server.log 2>&1 &
cd ..
echo "开始训练:"

python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_ppo.py ../../config/${model_name}/grpo_argument.json \
    --train_datasets "ppo-kk/34567ppl/train.jsonl" \
    --eval_datasets "ppo-kk/5ppl/test.jsonl" \
    --label_key tgt \
    --rl_algorithm reinforce_plus_plus \
    --normalize_advantage 0 \
    --normalize_reward 1 \
    --actor_model_name_or_path ${model_name_or_path} \
    --reward_model_name_or_path "" \
    --output_dir ${output_dir} \
    --max_steps ${steps} \
    --save_steps ${steps} \
    --tensor_parallel_degree 2 \
    --per_device_prompt_batch_size 1 \
    --per_device_train_batch_size 8 \
    --max_length 1024 \
    --max_prompt_len 512 \
    --pipeline_parallel_degree 1 \
    --sharding_parallel_degree 4 \
    --sharding "stage1" \
    --recompute 1 \
    ${ext_args}

echo "热启"
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_ppo.py ../../config/${model_name}/grpo_argument.json \
    --train_datasets "ppo-kk/34567ppl/train.jsonl" \
    --eval_datasets "ppo-kk/5ppl/test.jsonl" \
    --label_key tgt \
    --rl_algorithm reinforce_plus_plus \
    --normalize_advantage 0 \
    --normalize_reward 1 \
    --actor_model_name_or_path ${model_name_or_path} \
    --reward_model_name_or_path "" \
    --output_dir ${output_dir} \
    --max_steps 1 \
    --save_steps 11 \
    --tensor_parallel_degree 2 \
    --per_device_prompt_batch_size 1 \
    --per_device_train_batch_size 8 \
    --max_length 1024 \
    --max_prompt_len 512 \
    --pipeline_parallel_degree 1 \
    --sharding_parallel_degree 4 \
    --sharding "stage1" \
    --recompute 1 \
    ${ext_args}

echo "kill reward 服务"
pkill -9 -f reward_server.py
