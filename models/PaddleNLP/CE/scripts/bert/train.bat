@echo off
cd ../..
if not exist log\bert md log\bert
set logpath=%cd%\log\bert

cd models_repo\examples\language_model\bert\

python -m paddle.distributed.launch --gpus %2 run_pretrain.py --model_type bert --model_name_or_path bert-base-uncased --max_predictions_per_seq 20 --batch_size 32   --learning_rate 1e-4 --weight_decay 1e-2 --adam_epsilon 1e-6 --warmup_steps 10000 --num_train_epochs 1 --input_dir data/ --output_dir pretrained_models/ --logging_steps 1 --save_steps 1 --max_steps 1 --device %1 --use_amp False > %logpath%\train_win_%1.log 2>&1
