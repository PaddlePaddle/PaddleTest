@echo off
cd ../..

md log\waybill_ie
set logpath=%cd%\log\waybill_ie

cd models_repo\examples\information_extraction\waybill_ie\

if "%2"=="ernie" (
    python run_ernie.py --batch_size 20 --epochs 1 --device %1 > %logpath%\train_%2_%1.log 2>&1
) else if "%2"=="ernie_crf" (
    python run_ernie_crf.py --batch_size 20 --epochs 1 --device %1 > %logpath%\train_%2_%1.log 2>&1
) else (
    python run_bigru_crf.py --batch_size 20 --epochs 1 --device %1 > %logpath%\train_%2_%1.log 2>&1
)

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%\train_%2_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%\train_%2_%1.log
)
