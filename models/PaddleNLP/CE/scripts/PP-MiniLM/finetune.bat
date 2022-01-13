@echo off
cd ../..

if not exist log\PP-MiniLM md log\PP-MiniLM
set logpath=%cd%\log\PP-MiniLM

cd models_repo\examples\model_compression\pp-minilm\finetuning
md ppminilm-6l-768h

python -u ./run_clue.py --model_type ppminilm  --model_name_or_path ppminilm-6l-768h --task_name %3 --max_seq_length %6 --batch_size %5  --learning_rate %4 --num_train_epochs 1 --logging_steps 100 --seed 42  --save_steps  100 --warmup_proportion 0.1 --weight_decay 0.01 --adam_epsilon 1e-8 --output_dir ppminilm-6l-768h/models/%3/%4_%5/ --device %1 > %logpath%/finetune_%3_%4_%5_%1.log 2>&1

python export_model.py --task_name %3 --output_dir ./ppminilm-6l-768h/models/%3/%4_%5

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/finetune_%3_%4_%5_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/finetune_%3_%4_%5_%1.log
)
