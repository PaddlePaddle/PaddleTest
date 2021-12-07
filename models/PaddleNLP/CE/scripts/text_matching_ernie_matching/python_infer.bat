@echo off
cd ../..
if not exist log\text_matching_ernie_matching md log\text_matching_ernie_matching
set logpath=%cd%\log\text_matching_ernie_matching

cd models_repo\examples\text_matching\ernie_matching\

python export_model.py --params_path ./checkpoints/%3/model_30/model_state.pdparams --output_path=./output_%3
python deploy/python/predict.py --model_dir ./output_%3 --device %1 > %logpath%/python_infer_%3_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/python_infer_%3_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/python_infer_%3_%1.log
)
