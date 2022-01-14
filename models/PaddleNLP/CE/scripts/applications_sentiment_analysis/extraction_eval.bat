@echo off
cd ../..

if not exist log\applications_sentiment_analysis md log\applications_sentiment_analysis
set logpath=%cd%\log\applications_sentiment_analysis

cd models_repo\applications\sentiment_analysis\extraction\
python evaluate.py --model_path "../checkpoints/ext_checkpoints/best.pdparams" --test_path "../data/ext_data/test.txt" --label_path "../data/ext_data/label.dict" --batch_size 4 --max_seq_len 256 > %logpath%\extraction_eval_%1.log 2>&1
