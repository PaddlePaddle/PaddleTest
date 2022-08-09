@echo off
cd ../..

set logpath=%cd%\log\skep

cd models_repo\examples\sentiment_analysis\skep\

python predict_sentence.py --model_name "skep_ernie_1.0_large_ch" --device %1 --params_path checkpoints/model_2000/model_state.pdparams > %logpath%/infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
