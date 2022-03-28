@echo off
cd ../..

if not exist log\bert md log\bert
set logpath=%cd%\log\bert

cd models_repo\examples\language_model\bert\

python -m paddle.distributed.launch --gpus %2 run_glue.py --model_type bert --model_name_or_path bert-base-uncased --task_name SST2 --max_seq_length 128 --batch_size 32  --learning_rate 2e-5 --num_train_epochs 3 --logging_steps 1 --save_steps 1 --max_steps 1 --output_dir ./tmp/ --device %1 --use_amp False > %logpath%\finetune_win_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%\finetune_win_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%\finetune_win_%1.log
)
