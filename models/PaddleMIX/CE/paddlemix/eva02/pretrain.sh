#!/bin/bash

log_dir=${root_path}/log

exit_code=0

echo "*******paddlemix eva02 prtrain begin***********"

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree

optim="adamw"
lr=3e-3
warmup_lr=1e-6
min_lr=1e-5
weight_decay=0.05
CLIP_GRAD=3.0

num_train_epochs=1
save_epochs=1

warmup_epochs=1
drop_path=0.0

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'

TRAINERS_NUM=1           # nnodes, machine num
TRAINING_GPUS_PER_NODE=1 # nproc_per_node
DP_DEGREE=1              # dp_parallel_degree
MP_DEGREE=1              # tensor_parallel_degree
SHARDING_DEGREE=1        # sharding_parallel_degree

model_name="paddlemix/EVA/EVA02/eva02_Ti_for_pretrain"
teacher_name="paddlemix/EVA/EVA01-CLIP-g-14"
student_name="paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14"

TEA_PRETRAIN_CKPT=None
STU_PRETRAIN_CKPT=None

OUTPUT_DIR=./output/eva02_Ti_pt_in21k_p14

DATA_PATH=${root_path}/dataset/ILSVRC2012_tiny
input_size=224
num_mask_patches=105 ### 224*224/14/14 * 0.4
batch_size=2
num_workers=2
accum_freq=1     # update_freq
logging_steps=10 # print_freq
seed=0

USE_AMP=False
FP16_OPT_LEVEL="O1"
enable_tensorboard=True

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
(${TRAINING_PYTHON} run_eva02_pretrain_dist.py \
  --do_train \
  --data_path ${DATA_PATH}/train \
  --model ${model_name} \
  --teacher ${teacher_name} \
  --student ${student_name} \
  --input_size ${input_size} \
  --drop_path ${drop_path} \
  --optim ${optim} \
  --learning_rate ${lr} \
  --weight_decay ${weight_decay} \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-6 \
  --max_grad_norm ${CLIP_GRAD} \
  --lr_scheduler_type cosine \
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
  --stu_pretrained_model_path ${STU_PRETRAIN_CKPT} \
  --tea_pretrained_model_path ${TEA_PRETRAIN_CKPT} \
  --fp16_opt_level ${FP16_OPT_LEVEL} \
  --seed ${seed} \
  --recompute True \
  --bf16 ${USE_AMP}) 2>&1 | tee ${log_dir}/run_mix_eva02_pretrain.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
  echo "paddlemix eva02 pretrain run success" >>"${log_dir}/ce_res.log"
else
  echo "paddlemix eva02 pretrain run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix eva02 pretrain end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
  exit 1
fi
