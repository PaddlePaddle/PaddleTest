@echo off
cd ../..

md log\tinybert

set logpath=%cd%\log\tinybert

cd models_repo\examples\model_compression\tinybert\

cd ..\..\benchmark\glue\

python -u ./run_glue.py --model_type bert --model_name_or_path bert-base-uncased --task_name %2 --max_seq_length 128 --batch_size 32   --learning_rate 2e-5 --num_train_epochs 1 --logging_steps 10 --save_steps 10 --max_steps 30 --output_dir ./tmp/%2/ --device %1 > %logpath%/finetune_%2_%1.log
