@echo off
cd ../..

if not exist log\applications_sentiment_analysis md log\applications_sentiment_analysis
set logpath=%cd%\log\applications_sentiment_analysis

cd models_repo\applications\sentiment_analysis
python export_model.py --model_type "extraction" --model_path "./checkpoints/ext_checkpoints/best.pdparams" --save_path "./checkpoints/ext_checkpoints/static/infer"
python export_model.py --model_type "classification" --model_path "./checkpoints/cls_checkpoints/best.pdparams" --save_path "./checkpoints/cls_checkpoints/static/infer"
cd deploy
python predict.py --base_model_name "skep_ernie_1.0_large_ch" --ext_model_path "../checkpoints/ext_checkpoints/static/infer" --cls_model_path "../checkpoints/cls_checkpoints/static/infer" --ext_label_path "../data/ext_data/label.dict" --cls_label_path "../data/cls_data/label.dict" --test_path "../data/test.txt" --save_path "../data/sentiment_results.json" --batch_size 8 --ext_max_seq_len 512 --cls_max_seq_len 256 > %logpath%\python_infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/python_infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/python_infer_%1.log
)
