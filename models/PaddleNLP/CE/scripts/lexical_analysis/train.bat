@echo off
cd ../..

md log\lexical_analysis
set logpath=%cd%\log\lexical_analysis

cd models_repo\examples\lexical_analysis\

python train.py --data_dir .\lexical_analysis_dataset_tiny --model_save_dir ./save_dir --epochs 1 --batch_size 16 --device %1  > %logpath%/train_%1.log 2>&1
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%1.log
)
