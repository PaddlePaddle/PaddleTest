@echo off
cd ../..
if not exist log\text_matching_ernie_matching md log\text_matching_ernie_matching
set logpath=%cd%\log\text_matching_ernie_matching

cd models_repo\examples\text_matching\ernie_matching\

if "%3"=="point-wise" (
    python -u -m paddle.distributed.launch --gpus %2 predict_pointwise.py --device %1 --params_path "./checkpoints/%3/model_30/model_state.pdparams" --batch_size 32 --max_seq_length 64 --input_file 'test.tsv' > %logpath%/infer_%3_%1.log 2>&1
) else (
    python -u -m paddle.distributed.launch --gpus %2 predict_pairwise.py --device %1 --params_path "./checkpoints/%3/model_30/model_state.pdparams"--batch_size 32 --max_seq_length 64 --input_file 'test.tsv' > %logpath%/infer_%3_%1.log 2>&1
)

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%3_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%3_%1.log
)
