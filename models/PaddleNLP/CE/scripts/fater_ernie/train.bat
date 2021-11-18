@echo off
cd ../..

if not exist log\fater_ernie md log\fater_ernie

set logpath=%cd%\log\fater_ernie

cd models_repo\examples\experimental\faster_ernie\%2

python train.py --device %1 --save_dir checkpoints/ --batch_size 16 --max_seq_length 128 --epochs 1 > %logpath%/train_%2_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%2_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%2_%1.log
)

