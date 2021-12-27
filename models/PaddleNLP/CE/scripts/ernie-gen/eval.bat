@echo off
cd ../..

if not exist log\ernie-gen md log\ernie-gen

set logpath=%cd%\log\ernie-gen

cd models_repo\examples\text_generation\ernie-gen\

python -u ./eval.py --model_name_or_path ernie-1.0 --max_encode_len 24 --max_decode_len 72 --batch_size 48  --init_checkpoint tmp\model_1000\model_state.pdparams --device %1 > %logpath%/eval_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/eval_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/eval_%1.log
)
