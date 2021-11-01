@echo off
cd ../..

if not exist log\ernie-csc md log\ernie-csc

set logpath=%cd%\log\ernie-csc

cd models_repo\examples\text_correction\ernie-csc\

python predict_sighan.py --model_name_or_path ernie-1.0 --test_file sighan_test/sighan14/input.txt --batch_size 32 --ckpt_path checkpoints/best_model.pdparams --predict_file predict_sighan14.txt > %logpath%/infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)