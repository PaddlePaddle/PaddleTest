@echo off
cd ../..

set logpath=%cd%\log\tinybert

cd models_repo\model_zoo\tinybert\

set SHEET_NAME_LOWER=%2

for %%i in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do call set SHEET_NAME_LOWER=%%SHEET_NAME_LOWER:%%i=%%i%%

set TEACHER_DIR=..\..\examples\benchmark\glue\tmp\%2\%SHEET_NAME_LOWER%_ft_model_30.pdparams

python task_distill.py --model_type tinybert --student_model_name_or_path tinybert-6l-768d-v2 --task_name %2 --intermediate_distill --max_seq_length 64 --batch_size 32   --T 1 --teacher_model_type bert --teacher_path %TEACHER_DIR% --learning_rate 5e-5 --num_train_epochs 1 --logging_steps 10 --save_steps 10 --max_steps 30 --output_dir ./tmp/%2/ --device %1 > %logpath%/distill_%2_%1.log


if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/distill_%2_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/distill_%2_%1.log
)
