@echo off
cd ../..

if not exist log\ernie-csc md log\ernie-csc

set logpath=%cd%\log\ernie-csc

cd models_repo\examples\text_correction\ernie-csc\

python export_model.py --params_path checkpoints/best_model.pdparams --output_path ./infer_model/static_graph_params
python predict.py --model_file infer_model/static_graph_params.pdmodel --params_file infer_model/static_graph_params.pdiparams > %logpath%/infer_deploy_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_deploy_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_deploy_%1.log
)
