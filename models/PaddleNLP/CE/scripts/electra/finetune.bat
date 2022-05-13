@echo off
cd ../..

if not exist log\electra md log\electra

set logpath=%cd%\log\electra

cd models_repo\model_zoo\electra

python -u ./run_glue.py --model_type electra --model_name_or_path electra-small --task_name SST-2 --max_seq_length 128 --batch_size 32  --learning_rate 1e-4 --num_train_epochs 1 --logging_steps 10 --save_steps 10 --output_dir ./SST-2/ --max_steps 50 --device %1 > %logpath%/finetune_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/finetune_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/finetune_%1.log
)
