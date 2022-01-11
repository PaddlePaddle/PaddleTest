@echo off
cd ../..

xcopy /y /c /h /r .\scripts\applications_sentiment_analysis\test.txt  .\models_repo\applications\sentiment_analysis\data\

if not exist log\applications_sentiment_analysis md log\applications_sentiment_analysis
set logpath=%cd%\log\applications_sentiment_analysis

cd models_repo\applications\sentiment_analysis

python predict.py --ext_model_path "./checkpoints/ext_checkpoints/best.pdparams" --cls_model_path "./checkpoints/cls_checkpoints/best.pdparams" --test_path "./data/test.txt" --ext_label_path "./data/ext_data/label.dict" --cls_label_path "./data/cls_data/label.dict" --save_path "./data/sentiment_results.json" --batch_size 8 --ext_max_seq_len 512 --cls_max_seq_len 256 > %logpath%\infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)