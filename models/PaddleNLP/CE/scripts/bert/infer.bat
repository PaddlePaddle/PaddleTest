@echo off
cd ../..

if not exist log\bert md log\bert
set logpath=%cd%\log\bert

cd models_repo\examples\language_model\bert\


python -u ./export_model.py --model_type bert --model_path bert-base-uncased --output_path ./infer_model/model
python -u ./predict_glue.py --task_name SST-2 --model_type bert --model_path ./infer_model/model --batch_size 32 --max_seq_length 128 > %logpath%\bert_win_predict.log 2>&1


if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%\bert_win_predict.log
) else (
    echo "exit_code: 0.0" >> %logpath%\bert_win_predict.log
)
