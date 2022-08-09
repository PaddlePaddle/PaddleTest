@echo off
cd ../..
if not exist log\text_matching_ernie_matching md log\text_matching_ernie_matching
set logpath=%cd%\log\text_matching_ernie_matching

cd models_repo\examples\text_matching\ernie_matching\

if "%3"=="point-wise" (
    python -u -m paddle.distributed.launch --gpus %2 train_pointwise.py --device %1 --save_dir ./checkpoints/%3 --batch_size 32 --epochs 1 --save_step 10 --max_step 30 --learning_rate 2E-5 > %logpath%/train_%3_%1.log 2>&1
) else (
    python -u -m paddle.distributed.launch --gpus %2 train_pairwise.py --device %1 --save_dir ./checkpoints/%3 --batch_size 32  --margin 2 --epochs 1 --save_step 10 --max_step 30 --learning_rate 2E-5 > %logpath%/train_%3_%1.log 2>&1
)

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%3_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%3_%1.log
)
