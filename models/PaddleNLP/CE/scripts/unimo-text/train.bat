@echo off
cd ../..

if not exist log\unimo-text md log\unimo-text

set logpath=%cd%\log\unimo-text

cd models_repo\examples\text_generation\unimo-text\

if not exist log md log

python -m paddle.distributed.launch --gpus "0" --log_dir ./log  run_gen.py --dataset_name=dureader_qg --model_name_or_path=unimo-text-1.0 --save_dir=./unimo/checkpoints --logging_steps=100 --save_steps=100000 --epochs=1 --batch_size=8 --learning_rate=5e-5 --warmup_propotion=0.02 --weight_decay=0.01 --max_seq_len=512 --max_target_len=30 --do_train --do_predict --device=%1 > %logpath%\train_%1.log 2>&1
