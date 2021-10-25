@echo off
cd ../..

if not exist log\few_shot_pet md log\few_shot_pet

set logpath=%cd%\log\few_shot_pet

cd models_repo\examples\few_shot\pet\

python -u -m paddle.distributed.launch --gpus %2 pet.py --task_name %3 --device %1 --pattern_id 0 --save_dir "checkpoints/%3" --index 0 --batch_size 4 --learning_rate 1E-4 --epochs 1 --max_seq_length 512 --save_steps 100 --language_model "ernie-1.0" --rdrop_coef %4 > %logpath%/train_%3_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%3_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%3_%1.log
)
