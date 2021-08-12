@echo off
cd ../..

if not exist log\bigbird md log\bigbird
set logpath=%cd%\log\bigbird

cd models_repo\examples\language_model\bigbird\

python -m paddle.distributed.launch --gpus %2 --log_dir log  run_pretrain.py --model_name_or_path bigbird-base-uncased --input_dir "./data" --output_dir "output" --batch_size 4 --weight_decay 0.01 --learning_rate 1e-5 --max_steps 1 --save_steps 1 --logging_steps 1 --max_encoder_length 512 --max_pred_length 75 > %logpath%\train_%1.log 2>&1
