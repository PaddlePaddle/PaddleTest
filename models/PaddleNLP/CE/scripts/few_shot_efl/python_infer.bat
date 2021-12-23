@echo off
cd ../..

set logpath=%cd%\log\few_shot_efl

cd models_repo\examples\few_shot\efl\

md python_output\%3

python export_model.py --params_path=./checkpoints/%3/model_%4/model_state.pdparams --output_path=./python_output/%3

python deploy/python/predict.py --model_dir=./python_output/%3 --task_name %3 --batch_size 4 > %logpath%/python_infer_%3_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/python_infer_%3_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/python_infer_%3_%1.log
)
