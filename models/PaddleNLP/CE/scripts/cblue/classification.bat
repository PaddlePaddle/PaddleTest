@echo off
cd ../..
if not exist log\cblue md log\cblue
set logpath=%cd%\log\cblue

cd models_repo\examples\benchmark\cblue\
python -m paddle.distributed.launch --gpus %3 train_classification.py --dataset %2 --batch_size 16 --max_seq_length 96 --learning_rate 3e-5 --epochs 1 --max_steps 20 --save_steps 10 --logging_steps 10 --valid_steps 10 --save_dir ./checkpoint/%2 > %logpath%/classification_%2_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/classification_%2_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/classification_%2_%1.log
)
