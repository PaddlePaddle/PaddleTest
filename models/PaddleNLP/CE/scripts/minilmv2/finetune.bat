
@echo off
cd ../..

md log\minilmv2

set logpath=%cd%\log\minilmv2

cd models_repo\examples\model_compression\minilmv2\


python -u ./run_clue.py --model_type tinybert --model_name_or_path ./10w --task_name AFQMC --max_seq_length 128 --batch_size 16 --learning_rate 2e-5 --num_train_epochs 3 --logging_steps 100 --seed 42 --save_steps  100 --warmup_proportion 0.1 --weight_decay 0.01 --adam_epsilon 1e-8 --device %1 > %log_path%/finetune_AFQMC_%2_%1.log
