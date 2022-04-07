@echo off
cd ../..
if not exist log\cblue md log\cblue
set logpath=%cd%\log\cblue

cd models_repo\examples\biomedical\cblue\

python export_model.py --train_dataset CHIP-STS --params_path=./checkpoint/CHIP-STS/model_10/model_state.pdparams --output_path=./export > %logpath%/infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
