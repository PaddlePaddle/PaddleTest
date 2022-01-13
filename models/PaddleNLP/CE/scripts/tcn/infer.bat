@echo off
cd ../..
if not exist log\tcn md log\tcn
set logpath=%cd%\log\tcn
cd models_repo\examples\time_series\tcn
python predict.py --data_path time_series_covid19_confirmed_global.csv --use_gpu > %logpath%/infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
