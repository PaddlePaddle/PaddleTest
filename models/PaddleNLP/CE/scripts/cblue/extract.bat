@echo off
cd ../..
if not exist log\cblue md log\cblue
set logpath=%cd%\log\cblue

cd models_repo\model_zoo\ernie-health\cblue\

python -m paddle.distributed.launch --gpus %2 train_spo.py --batch_size 4 --max_seq_length 300 --learning_rate 6e-5 --epochs 1 --max_steps 10 --save_steps 10 --logging_steps 1 --valid_steps 100 --save_dir ./checkpoint/CMeIE > %logpath%/extract_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/extract_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/extract_%1.log
)
