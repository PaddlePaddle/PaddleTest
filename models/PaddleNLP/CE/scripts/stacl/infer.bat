@echo off
cd ../..

set logpath=%cd%\log\stacl

cd models_repo\examples\simultaneous_translation\stacl\

python predict.py --config ./config/transformer.yaml > %logpath%/infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
