@echo off
cd ../..

if not exist log\ernie-ctm md log\ernie-ctm

set logpath=%cd%\log\ernie-ctm

cd models_repo\examples\text_to_knowledge\ernie-ctm\
python -m paddle.distributed.launch --gpus %2  train.py --max_seq_len 128 --batch_size 8 --learning_rate 5e-5 --num_train_epochs 1 --logging_steps 10 --save_steps 100 --output_dir ./tmp/ --device %1 > %logpath%/train_%1.log 2>&1
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%1.log
)
