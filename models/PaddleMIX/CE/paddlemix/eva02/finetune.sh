#!/bin/bash

log_dir=${root_path}/log

exit_code=0

echo "*******paddlemix eva02 finetune begin***********"

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree

optim="adamw"
lr=2e-4
layer_decay=0.9
warmup_lr=0.0
min_lr=0.0
weight_decay=0.05
CLIP_GRAD=0.0

num_train_epochs=1
save_epochs=1

warmup_epochs=0
warmup_steps=0
drop_path=0.1

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'

TRAINERS_NUM=1           # nnodes, machine num
TRAINING_GPUS_PER_NODE=1 # nproc_per_node
DP_DEGREE=2              # dp_parallel_degree
MP_DEGREE=1              # tensor_parallel_degree
SHARDING_DEGREE=2        # sharding_parallel_degree

MODEL_NAME="paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14"
PRETRAIN_CKPT=./model_state.pdparams

OUTPUT_DIR=./output/eva02_Ti_pt_in21k_ft_in1k_p14

DATA_PATH=/home/dataset/ILSVRC2012_tiny

input_size=336
batch_size=128
num_workers=2
accum_freq=1     # update_freq
logging_steps=10 # print_freq
seed=0

USE_AMP=False
FP16_OPT_LEVEL="O1"
enable_tensorboard=True

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
(${TRAINING_PYTHON} run_eva02_finetune_dist.py \
  --do_train \
  --data_path ${DATA_PATH}/train \
  --eval_data_path ${DATA_PATH}/val \
  --pretrained_model_path ${PRETRAIN_CKPT} \
  --model ${MODEL_NAME} \
  --input_size ${input_size} \
  --layer_decay ${layer_decay} \
  --drop_path ${drop_path} \
  --optim ${optim} \
  --learning_rate ${lr} \
  --weight_decay ${weight_decay} \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_epsilon 1e-8 \
  --max_grad_norm ${CLIP_GRAD} \
  --lr_scheduler_type cosine \
  --lr_end 1e-7 \
  --warmup_lr ${warmup_lr} \
  --min_lr ${min_lr} \
  --num_train_epochs ${num_train_epochs} \
  --save_epochs ${save_epochs} \
  --warmup_epochs ${warmup_epochs} \
  --per_device_train_batch_size ${batch_size} \
  --dataloader_num_workers ${num_workers} \
  --output_dir ${OUTPUT_DIR} \
  --logging_dir ${OUTPUT_DIR}/tb_log \
  --logging_steps ${logging_steps} \
  --accum_freq ${accum_freq} \
  --dp_degree ${DP_DEGREE} \
  --tensor_parallel_degree ${MP_DEGREE} \
  --sharding_parallel_degree ${SHARDING_DEGREE} \
  --pipeline_parallel_degree 1 \
  --disable_tqdm True \
  --tensorboard ${enable_tensorboard} \
  --recompute True \
  --fp16_opt_level ${FP16_OPT_LEVEL} \
  --seed ${seed} \
  --fp16 ${USE_AMP}) 2>&1 | tee ${log_dir}/run_mix_eva02_finetune.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "paddlemix eva02 finetune run success" >>"${log_dir}/ce_res.log"
else
  echo "paddlemix eva02 finetune run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix eva02 finetune end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
  exit 1
fi
