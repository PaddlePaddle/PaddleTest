@echo off
cd ../..

if not exist log\bigbird md log\bigbird
set logpath=%cd%\log\bigbird

cd models_repo\examples\language_model\bigbird\

python run_classifier.py --model_name_or_path ./output/model_10 --output_dir "output" --batch_size 1 --learning_rate 5e-6 --max_steps 10 --save_steps 10 --logging_steps 1 --max_encoder_length 576 > %logpath%\finetune_%1.log 2>&1
