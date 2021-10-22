@echo off
cd ../..

if not exist log\language_model_ernie_doc md log\language_model_ernie_doc

set logpath=%cd%\log\language_model_ernie_doc

cd models_repo\examples\language_model\ernie-doc\

python run_sequence_labeling.py --batch_size 4 --learning_rate 3e-5 --model_name_or_path %2 --dataset %3 --save_steps 10 --max_steps 30 --logging_steps 10 --device %1 > %logpath%\train_%3_%1.log 2>&1
