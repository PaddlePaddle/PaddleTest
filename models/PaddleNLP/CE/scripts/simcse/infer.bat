@echo off
cd ../..
if not exist log\simcse md log\simcse

set logpath=%cd%\log\simcse

cd models_repo\examples\text_matching\simcse\

python -u -m paddle.distributed.launch --gpus %2 predict.py --device %1 --params_path "./%3/model_%4/model_state.pdparams" --batch_size 64 --max_seq_length 64 --text_pair_file "./senteval_cn/%3/test.txt" > %logpath%/infer_%3_%1.log 2>&1
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%3_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%3_%1.log
)
