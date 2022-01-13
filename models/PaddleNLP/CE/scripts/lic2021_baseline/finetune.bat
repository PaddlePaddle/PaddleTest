@echo off
cd ../..

if not exist log\lic2021_baseline md log\lic2021_baseline

set logpath=%cd%\log\lic2021_baseline

cd models_repo\examples\dialogue\lic2021_baseline\

md log
python -m paddle.distributed.launch --gpus %2 --log_dir ./log finetune.py --model_name_or_path=unified_transformer-12L-cn --train_data_path=./datasets/train.txt --valid_data_path=./datasets/valid.txt --save_dir=./checkpoints --logging_steps=500 --save_steps=8000 --seed=2021 --epochs=1 --batch_size=2048 --lr=1e-5 --weight_decay=0.01 --warmup_steps=4000 --max_grad_norm=0.1 --sort_pool_size=65536 --device=%1 > %logpath%/finetune_%1.log
