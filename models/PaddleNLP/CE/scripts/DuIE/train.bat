@echo off
cd ../..

if not exist log\DuIE md log\DuIE

set logpath=%cd%\log\DuIE

cd models_repo\examples\information_extraction\DuIE\

python -m paddle.distributed.launch --gpus %2 run_duie.py --device %1 --seed 42 --do_train --data_path ./data --max_seq_length 128 --batch_size 8 --num_train_epochs 1 --learning_rate 2e-5 --warmup_ratio 0.06 --output_dir ./checkpoints > %logpath%/train_%1.log 2>&1
