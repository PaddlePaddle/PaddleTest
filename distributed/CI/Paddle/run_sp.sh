set -x
set -e
export PYTHONPATH=/workspace/PaddleNLP/:$PYTHONPATH
export FLAGS_infer_spmd_enable=true  
export FLAGS_call_stack_level=2

task="sp_acc_check"
mp_degree=2
dp_degree=1
pp_degree=1
local_batch_size=1

# sp on
sp=True
rm -rf ./${task}_mp${mp_degree}_sp${sp}/*

python -m paddle.distributed.launch --log_dir=./${task}_mp${mp_degree}_sp${sp} --devices=0,1 --rank 0 tools/auto.py \
    -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp8.yaml \
    -o Model.hidden_size=1024 \
    -o Model.num_layers=12 \
    -o Model.hidden_dropout_prob=0 \
    -o Model.attention_probs_dropout_prob=0 \
    -o Model.use_recompute=True \
    -o Optimizer.grad_clip.clip_norm=0 \
    -o Global.local_batch_size=$(($local_batch_size / $dp_degree)) \
    -o Global.micro_batch_size=$(($local_batch_size / $dp_degree)) \
    -o Distributed.dp_degree=${dp_degree} \
    -o Distributed.mp_degree=${mp_degree} \
    -o Distributed.pp_degree=${pp_degree} \
    -o Distributed.sharding.sharding_degree=1 \
    -o Distributed.sharding.sharding_stage=1 \
    -o Distributed.schedule_mode=FThenB \
    -o Engine.mix_precision.enable=False \
    -o Engine.mix_precision.level=o2 \
    -o Engine.max_steps=30 \
    -o Engine.eval_freq=100000 \
    -o Engine.verbose=3 \
    -o Engine.logging_freq=1 \
    -o Engine.save_load.output_dir="" \
    -o Model.sequence_parallel=${sp}

mv *_program_* ./${task}_mp${mp_degree}_sp${sp}

# sp off
sp=False
rm -rf ./${task}_mp${mp_degree}_sp${sp}/*

python -m paddle.distributed.launch --log_dir=./${task}_mp${mp_degree}_sp${sp} --devices=0,1 --rank 0 tools/auto.py \
    -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp8.yaml \
    -o Model.hidden_size=1024 \
    -o Model.num_layers=12 \
    -o Model.hidden_dropout_prob=0 \
    -o Model.attention_probs_dropout_prob=0 \
    -o Model.use_recompute=True \
    -o Optimizer.grad_clip.clip_norm=0 \
    -o Global.local_batch_size=$(($local_batch_size / $dp_degree)) \
    -o Global.micro_batch_size=$(($local_batch_size / $dp_degree)) \
    -o Distributed.dp_degree=${dp_degree} \
    -o Distributed.mp_degree=${mp_degree} \
    -o Distributed.pp_degree=${pp_degree} \
    -o Distributed.sharding.sharding_degree=1 \
    -o Distributed.sharding.sharding_stage=1 \
    -o Distributed.schedule_mode=FThenB \
    -o Engine.mix_precision.enable=False \
    -o Engine.mix_precision.level=o2 \
    -o Engine.max_steps=30 \
    -o Engine.eval_freq=100000 \
    -o Engine.verbose=3 \
    -o Engine.logging_freq=1 \
    -o Engine.save_load.output_dir="" \
    -o Model.sequence_parallel=${sp}

mv *_program_* ./${task}_mp${mp_degree}_sp${sp}


set +e
set +x

# sed -n '175, 300p' ./mylog/workerlog.0 | grep 'persist' | awk '{if ($0 ~ /param/) print $4; else print $3}' > startup_persist_var.csv
# rm -f ./ppfleetx/data/data_tools/cpp/fast_index_map_helpers.so


