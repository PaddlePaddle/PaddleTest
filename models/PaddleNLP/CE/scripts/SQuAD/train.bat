@echo off
cd ../..

if not exist log\SQuAD md log\SQuAD

set logpath=%cd%\log\SQuAD

cd models_repo\examples\machine_reading_comprehension\SQuAD\

python -m paddle.distributed.launch --gpus %2  run_squad.py --model_type bert --model_name_or_path bert-base-uncased --max_seq_length 384 --batch_size 12 --learning_rate 3e-5 --num_train_epochs 1 --max_steps 1 --logging_steps 1 --save_steps 1 --warmup_proportion 0.1 --weight_decay 0.01 --output_dir ./tmp/squad/  --do_train --do_predic --device=%1  > %logpath%\train_win_%1.log 2>&1