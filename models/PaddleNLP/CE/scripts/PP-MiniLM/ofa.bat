@echo off
cd ../..

if not exist log\PP-MiniLM md log\PP-MiniLM
set logpath=%cd%\log\PP-MiniLM
set output_path=%cd%\models_repo\examples\model_compression\pp-minilm\finetuning\ppminilm-6l-768h\

cd models_repo\examples\model_compression\pp-minilm\pruning

python -u ./prune.py --model_type ppminilm --model_name_or_path %output_path%/models/%3/%4_%5 --task_name %3 --max_seq_length %6 --batch_size %5 --learning_rate %4 --num_train_epochs 1 --logging_steps 100 --save_steps 100 --output_dir ./pruned_models/%3/0.75/best_model --device %1 --width_mult_list 0.75 > %logpath%/ofa_%3_%4_%5_%1.log 2>&1

python export_model.py --model_type ppminilm --task_name %3 --model_name_or_path pruned_models/%3/0.75/best_model --sub_model_output_dir pruned_models/%3/0.75/sub/  --static_sub_model pruned_models/%3/0.75/sub_static/float  --n_gpu 1 --width_mult 0.75


if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/ofa_%3_%4_%5_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/ofa_%3_%4_%5_%1.log
)
