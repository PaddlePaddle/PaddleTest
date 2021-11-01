@echo off
cd ../..

if not exist log\ernie-csc md log\ernie-csc

set logpath=%cd%\log\ernie-csc

cd models_repo\examples\text_correction\ernie-csc\

python sighan_evaluate.py -p predict_sighan14.txt -t sighan_test/sighan14/truth.txt > %logpath%/eval_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/eval_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/eval_%1.log
)