
@echo off
cd ../..

if not exist log\minilmv2 md log\minilmv2

set logpath=%cd%\log\minilmv2

cd models_repo\examples\model_compression\minilmv2\

python general_distill.py --student_model_type tinybert --num_relation_heads 48 --student_model_name_or_path tinybert-6l-768d-zh --init_from_student False --teacher_model_type bert --teacher_model_name_or_path bert-base-chinese --max_seq_length 128 --batch_size 4 --learning_rate 6e-4 --logging_steps 10 --max_steps 20 --warmup_steps 4000 --save_steps 10 --teacher_layer_index 11 --student_layer_index 5 --weight_decay 1e-2 --output_dir ./pretrain --device %1 --input_dir ./data > %logpath%/finetune_%1.log