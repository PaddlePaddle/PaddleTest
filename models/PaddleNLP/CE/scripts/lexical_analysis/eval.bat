@echo off
cd ../..
set logpath=%cd%\log\lexical_analysis
cd models_repo\examples\lexical_analysis\

python eval.py --data_dir ./lexical_analysis_dataset_tiny --init_checkpoint ./save_dir/model_62.pdparams --batch_size 16 --device %1 > %logpath%/eval_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/eval_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/eval_%1.log
)
