@echo off
cd ../..

if not exist log\ernie-gen md log\ernie-gen

set logpath=%cd%\log\ernie-gen

cd models_repo\examples\text_generation\ernie-gen\

python -u ./train.py --model_name_or_path ernie-1.0 --max_encode_len 12 --max_decode_len 72 --batch_size 12  --learning_rate 2e-5 --num_epochs 1 --logging_steps 10 --save_steps 1000 --output_dir ./tmp/ --device %1 > %logpath%/train_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%1.log
)
