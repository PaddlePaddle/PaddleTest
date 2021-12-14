@echo off
cd ../..

if not exist log\electra md log\electra

set logpath=%cd%\log\electra

cd models_repo\examples\language_model\electra

python -u ./export_model.py --input_model_dir ./SST-2/sst-2_ft_model_40.pdparams/ --output_model_dir ./ --model_name electra-deploy

python -u ./deploy/python/predict.py --model_file ./electra-deploy.pdmodel --params_file ./electra-deploy.pdiparams --predict_sentences "uneasy mishmash of styles and genres ." "director rob marshall went out gunning to make a great one ." --batch_size 2 --max_seq_length 128 --model_name electra-small > %logpath%/infer.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer.log
)
