@echo off
cd ../..
if not exist log\simbert md log\simbert

set logpath=%cd%\log\simbert

cd models_repo\examples\text_matching\simbert\

python predict.py --input_file .\datasets\lcqmc\dev.tsv --device %1 > %logpath%\infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
