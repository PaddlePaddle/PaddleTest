@echo off
cd ../..

if not exist log\few_shot_efl md log\few_shot_efl

set logpath=%cd%\log\few_shot_efl

cd models_repo\examples\few_shot\efl\

if not exist predict_output md predict_output

python -u -m paddle.distributed.launch --gpus %2 train.py --task_name %3 --device %1 --negative_num 1 --save_dir "checkpoints/%3" --batch_size 4 --learning_rate 5E-5 --epochs 1 --max_seq_length 512 --save_steps 100 > %logpath%/train_%3_%1.log 2>&1
