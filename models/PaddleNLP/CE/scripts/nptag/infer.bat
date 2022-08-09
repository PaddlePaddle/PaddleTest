@echo off
cd ../..

if not exist log\nptag md log\nptag

set logpath=%cd%\log\nptag

cd models_repo\examples\text_to_knowledge\nptag

python -m paddle.distributed.launch --gpus %2 predict.py --device=%1 --params_path ./output/model_100/model_state.pdparams > %logpath%/infer_%1.log 2>&1
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
