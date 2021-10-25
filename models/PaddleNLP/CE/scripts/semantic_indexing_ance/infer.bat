@echo off
cd ../..

set logpath=%cd%\log\semantic_indexing_ance

cd models_repo\examples\semantic_indexing\

if "%3"=="batch" (
    python -u -m paddle.distributed.launch --gpus %2 predict.py --device %1 --params_path "./checkpoints_batch_neg/model_1000/model_state.pdparams" --output_emb_size 256 --batch_size 32 --max_seq_length 64 --text_pair_file semantic_pair_train.tsv > %logpath%\infer_%3_%1.log 2>&1
) else (
    python -u -m paddle.distributed.launch --gpus %2 predict.py  --device %1 --params_path "./checkpoints_hardest_neg/model_1000/model_state.pdparams" --output_emb_size 256 --batch_size 32 --max_seq_length 64 --text_pair_file  semantic_pair_train.tsv > %logpath%\infer_%3_%1.log 2>&1
)

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%3_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%3_%1.log
)
