@echo off
cd ../..

if not exist log\ernie-ctm md log\ernie-ctm

set logpath=%cd%\log\ernie-ctm
cd models_repo\examples\text_to_knowledge\ernie-ctm\
python -m paddle.distributed.launch --gpus %2 predict.py --params_path ./tmp/model_100/model_state.pdparams --batch_size 16 --device %1 > %logpath%/infer_%1.log 2>&1
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
