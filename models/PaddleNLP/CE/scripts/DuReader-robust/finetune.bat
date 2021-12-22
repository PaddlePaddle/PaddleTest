@echo off
cd ../..

if not exist log\DuReader-robust md log\DuReader-robust

set logpath=%cd%\log\DuReader-robust

cd models_repo\examples\machine_reading_comprehension\DuReader-robust

python -m paddle.distributed.launch --gpus %2 run_du.py --task_name dureader_robust --model_type bert --model_name_or_path bert-base-chinese --max_seq_length 384 --batch_size 4 --learning_rate 3e-5 --num_train_epochs 1 --logging_steps 10 --save_steps 10 --max_steps 20 --warmup_proportion 0.1 --weight_decay 0.01 --output_dir ./tmp/dureader-robust/ --do_train --do_predict --device %1 > %logpath%/finetune_%1.log 2>&1
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/finetune_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/finetune_%1.log
)
