@echo off
cd ../..

if not exist log\applications_sentiment_analysis md log\applications_sentiment_analysis
set logpath=%cd%\log\applications_sentiment_analysis

cd models_repo\applications\sentiment_analysis\

python export_model.py --model_type "pp_minilm" --base_model_name "ppminilm-6l-768h" --model_path "./checkpoints/pp_checkpoints/best.pdparams" --save_path "./checkpoints/pp_checkpoints/static/infer"

cd pp_minilm

python quant_post.py --base_model_name "ppminilm-6l-768h" --static_model_dir "../checkpoints/pp_checkpoints/static" --quant_model_dir "../checkpoints/pp_checkpoints/quant" --algorithm "avg" --dev_path "../data/cls_data/dev.txt" --label_path "../data/cls_data/label.dict" --batch_size 4 --max_seq_len 256 --save_model_filename "infer.pdmodel" --save_params_filename "infer.pdiparams" --input_model_filename "infer.pdmodel" --input_param_filename "infer.pdiparams"

python performance_test.py --base_model_name "ppminilm-6l-768h" --model_path "../checkpoints/pp_checkpoints/quant/infer" --test_path "../data/cls_data/test.txt" --label_path "../data/cls_data/label.dict" --batch_size 16 --max_seq_len 256 --eval > %logpath%\quant_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/quant_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/quant_%1.log
)