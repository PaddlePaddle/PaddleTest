@echo off
cd ../..

if not exist log\ernie-gen md log\ernie-gen

set logpath=%cd%\log\ernie-gen

cd models_repo\model_zoo\ernie-gen\

python -u ./predict.py --model_name_or_path ernie-1.0 --max_encode_len 24 --max_decode_len 72 --batch_size 4 --init_checkpoint tmp\model_30\model_state.pdparams --device %1 > %logpath%/infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
