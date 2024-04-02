@echo off
cd ../..
md log\xlnet
set logpath=%cd%\log\xlnet

cd models_repo\examples\language_model\xlnet\

python -m paddle.distributed.launch --gpus %2 ./run_glue.py --model_name_or_path xlnet-base-cased --task_name %1 --max_seq_length 128 --batch_size 8 --learning_rate 2e-5 --num_train_epochs 1 --logging_steps 1 --save_steps 10 --max_steps 10 --output_dir .\$1\ > %logpath%\%1-fine_tune.log 2>&1
