@echo off
cd ../..

if not exist log\gpt md log\gpt

set logpath=%cd%\log\gpt

cd models_repo\examples\language_model\gpt\

xcopy /e /y /c /h /r D:\ce_data\paddleNLP\gpt2\  .\

python -m paddle.distributed.launch --gpus %2 run_pretrain.py --model_type gpt --model_name_or_path gpt2-en --input_dir "./data" --output_dir "output" --weight_decay 0.01 --grad_clip 1.0 --max_steps 1 --save_steps 1 --decay_steps 1 --warmup_rate 0.01 --micro_batch_size 2 --device %1  > %logpath%\train_gpt2_%1.log 2>&1
