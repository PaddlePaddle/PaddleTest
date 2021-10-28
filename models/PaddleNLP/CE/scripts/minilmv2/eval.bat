
@echo off
cd ../..

if not exist log\minilmv2 md log\minilmv2

set logpath=%cd%\log\minilmv2

cd models_repo\examples\model_compression\minilmv2\

python -u ./run_clue.py --model_type tinybert --model_name_or_path ./minilmv2_6l_768d_ch --task_name %2 --max_seq_length 128 --batch_size 4 --learning_rate 2e-5 --num_train_epochs 1 --logging_steps 1 --seed 42 --save_steps 10 --max_steps 10  --warmup_proportion 0.1 --weight_decay 0.01 --adam_epsilon 1e-8 --device %1 > %logpath%/eval_%2_%1.log
