@echo off
cd ../..
if not exist log\cblue md log\cblue
set logpath=%cd%\log\cblue

cd models_repo\examples\biomedical\cblue\

python -m paddle.distributed.launch --gpus %2 train_ner.py --batch_size 32 --max_seq_length 128 --learning_rate 6e-5 --epochs 1 --max_steps 20 --save_steps 10 --logging_steps 10 --valid_steps 10 --save_dir ./checkpoint/CMeEE > %logpath%/identify_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/identify_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/identify_%1.log
)
