@echo off
cd ../..

md log\electra

set logpath=%cd%\log\electra

cd models_repo\examples\language_model\electra

set DATA_DIR=%cd%\BookCorpus\

python -u ./run_pretrain.py --model_type electra --model_name_or_path electra-small --input_dir %DATA_DIR% --output_dir ./pretrain_model/ --train_batch_size 32 --learning_rate 5e-4 --max_seq_length 128 --weight_decay 1e-2 --adam_epsilon 1e-6 --warmup_steps 10000 --num_train_epochs 1 --logging_steps 1 --save_steps 10 --device %1 --max_steps 30 > %logpath%\train_%1.log 2>&1
