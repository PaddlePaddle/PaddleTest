@echo off
cd ../..
md log\glue
set logpath=%cd%\log\glue

cd models_repo\examples\benchmark\glue\

python -u .\run_glue.py --model_type %2 --model_name_or_path %3 --task_name  %4 --max_seq_length 128 --batch_size 16  --learning_rate %5 --num_train_epochs 1  --logging_steps 1 --save_steps 10 --output_dir .\%4 --max_steps 10 --device %1  > %logpath%\train_%3_%4_%1.log 2>&1
