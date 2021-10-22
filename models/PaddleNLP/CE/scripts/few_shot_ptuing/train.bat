@echo off
cd ../..

if not exist log\few_shot_ptuing md log\few_shot_ptuing

set logpath=%cd%\log\few_shot_ptuing

cd models_repo\examples\few_shot\p-tuning\

if not exist predict_output md predict_output

if "%3"=="iflytek" (
    python -u -m paddle.distributed.launch --gpus %2 ptuning.py --task_name %3 --device %1 --p_embedding_num 1 --save_dir "checkpoints/%3" --batch_size 8 --learning_rate 1E-4 --epochs 1 --save_steps 20 --max_seq_length 512 > %logpath%/train_%3_%1.log 2>&1
) else if "%3"=="csldcp" (
    python -u -m paddle.distributed.launch --gpus %2 ptuning.py --task_name %3 --device %1 --p_embedding_num 2 --save_dir "checkpoints/%3" --batch_size 8 --learning_rate 1E-4 --epochs 1 --save_steps 20 --max_seq_length 512 > %logpath%/train_%3_%1.log 2>&1
) else (
    python -u -m paddle.distributed.launch --gpus %2 ptuning.py --task_name %3 --device %1 --p_embedding_num 1 --save_dir "checkpoints/%3" --batch_size 4 --learning_rate 5E-5 --epochs 1 --save_steps 20 --max_seq_length 512 > %logpath%/train_%3_%1.log 2>&1
)

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%3_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%3_%1.log
)
