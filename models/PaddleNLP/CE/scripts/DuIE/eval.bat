@echo off
cd ../..

set logpath=%cd%\log\DuIE

cd models_repo\examples\information_extraction\DuIE\

python re_official_evaluation.py --golden_file=./data/dev_data.json  --predict_file=./data/predictions.json.zip > %logpath%/eval_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/eval_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/eval_%1.log
)
