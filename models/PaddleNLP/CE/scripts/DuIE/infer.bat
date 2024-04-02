@echo off
cd ../..

set logpath=%cd%\log\DuIE

cd models_repo\examples\information_extraction\DuIE\

python run_duie.py --do_predict --init_checkpoint ./checkpoints/model_10000.pdparams --predict_data_file ./data/test.json --max_seq_length 128 --batch_size 64 > %logpath%/infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
