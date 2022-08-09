@echo off
cd ../..

if not exist log\skep md log\skep

set logpath=%cd%\log\skep

cd models_repo\examples\sentiment_analysis\skep\

python export_model.py --model_name="skep_ernie_1.0_large_ch" --params_path=./checkpoint/model_2000/model_state.pdparams --output_path=./static_graph_params  >> %logpath%/python_infer_%1.log 2>&1
python deploy/python/predict.py --model_name="skep_ernie_1.0_large_ch" --model_file=static_graph_params.pdmodel --params_file=static_graph_params.pdiparams >> %logpath%/python_infer_%1.log 2>&1
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/python_infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/python_infer_%1.log
)
