@echo off
cd ../..

if not exist log\DuReader-yesno md log\DuReader-yesno

set logpath=%cd%\log\DuReader-yesno

cd models_repo\examples\machine_reading_comprehension\DuReader-yesno\

python -m paddle.distributed.launch --gpus %2 run_du.py --model_type ernie_gram --model_name_or_path ernie-gram-zh --max_seq_length 384 --batch_size 6 --learning_rate 3e-5 --num_train_epochs 1  --logging_steps 10 --save_steps 10 --max_steps 20 --warmup_proportion 0.1 --weight_decay 0.01 --output_dir ./tmp/dureader-yesno/ --device %1 > %logpath%/finetune_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/finetune_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/finetune_%1.log
)
