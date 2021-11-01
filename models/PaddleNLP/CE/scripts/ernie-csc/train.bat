@echo off
cd ../..

if not exist log\ernie-csc md log\ernie-csc

set logpath=%cd%\log\ernie-csc

cd models_repo\examples\text_correction\ernie-csc\

python train.py --batch_size 4 --logging_steps 1 --epochs 1 --save_steps 10 --max_steps 30 --learning_rate 5e-5 --model_name_or_path ernie-1.0 --output_dir ./checkpoints/ --extra_train_ds_dir ./extra_train_ds/ --max_seq_length 192 --device %1 > %logpath%/train_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%1.log
)
