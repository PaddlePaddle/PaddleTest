
@echo off
cd ../..

if not exist log\unified_transformer md log\unified_transformer

set logpath=%cd%\log\unified_transformer

cd models_repo\examples\dialogue\unified_transformer\

if not exist log md log

python -m paddle.distributed.launch --gpus %2 --log_dir ./log finetune.py --model_name_or_path=unified_transformer-12L-cn-luge --save_dir=./checkpoints --logging_steps=100 --save_steps=1000 --seed=2021 --epochs=1 --batch_size=16 --lr=5e-5 --weight_decay=0.01 --warmup_steps=2500 --max_grad_norm=0.1 --max_seq_len=512 --max_response_len=128 --max_knowledge_len=256 --device=%1 >%logpath%\finetune_%1.log 2>&1
