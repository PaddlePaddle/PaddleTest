@echo off
cd ../..

if not exist log\nptag md log\nptag

set logpath=%cd%\log\nptag

cd models_repo\examples\text_to_knowledge\nptag

python -m paddle.distributed.launch --gpus %2 train.py --batch_size 64 --learning_rate 1e-6 --num_train_epochs 1 --logging_steps 10 --save_steps 100 --output_dir ./output --device %1> %logpath%/train_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%1.log
)
