@echo off
cd ../..

if not exist log\waybill_ie md log\waybill_ie

set logpath=%cd%\log\waybill_ie

cd models_repo\examples\information_extraction\waybill_ie\

python export_model.py --params_path %2_ckpt/model_80/model_state.pdparams --output_path=./%2_output  >> %logpath%/python_infer_%2_%1.log 2>&1
python deploy/python/predict.py --model_dir ./%2_output --batch_size 1 >> %logpath%/python_infer_%2_%1.log 2>&1
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/python_infer_%2_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/python_infer_%2_%1.log
)
