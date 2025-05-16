# work_path: PaddleLLM/llm/alignment/rl
# grpo 训练
model_name=$1
ngpus=${2:-0,1,2,3}
steps=${3:-1200}
ext_args=""

# 1. 模型准备
echo "清理显存"
# fuser -v /dev/nvidia* 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) print $i}' | xargs kill -9 2>/dev/null
sleep 3s
echo "清理Checkpoints"
rm -rf ../../checkpoints/${model_name}/grpo/* 2>/dev/null 

if [[ ${model_name} == "qwen" ]]; then
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct-1M"
elif [[ ${model_name} == "llama" ]]; then
    model_name_or_path="meta-llama/Meta-Llama-3-8B"
fi

output_dir="../../checkpoints/${model_name}/grpo" # 以llm为根目录

# 2. 数据准备 
if [ ! -d "ppo-kk" ]; then
    wget -q https://paddlenlp.bj.bcebos.com/datasets/examples/ppo-kk.tgz && tar zxf ppo-kk.tgz
fi

# 3. 设置环境变量
current_path=$(pwd)
repo_path=${current_path%%PaddleLLM*}PaddleLLM
llm_path=${repo_path}/llm
export PYTHONPATH=$repo_path:$PYTHONPATH
export PYTHONPATH=$llm_path:$PYTHONPATH

export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False
export HF_DATASETS_DOWNLOAD_TIMEOUT=1
export FLAGS_gemm_use_half_precision_compute_type=False
export FLAGS_force_cublaslt_no_reduced_precision_reduction=True

export FLAGS_mla_use_tensorcore=0
export FLAGS_cascade_attention_max_partition_size=2048

# 4. 启动训练脚本
if ! pgrep -f reward_server.py > /dev/null; then
    echo "reward服务未运行"
    cd reward
    nohup python reward_server.py > reward_server.log 2>&1 &
    sleep 60s
    cd ..
else
    echo "reward服务已运行"
fi

echo "开始训练:"
python -u -m paddle.distributed.launch --devices "$ngpus" --log_dir  "log_grpo" \
    run_rl.py ../../config/${model_name}/grpo_argument.yaml \
    --use_fused_rms_norm true \
    --actor_model_name_or_path ${model_name_or_path} \
    --output_dir ${output_dir} \
    --max_steps ${steps} \
    --save_steps ${steps} \
    --eval_steps  ${steps} \
    ${ext_args}
