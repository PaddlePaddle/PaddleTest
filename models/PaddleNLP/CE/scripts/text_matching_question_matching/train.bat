@echo off
cd ../..

if not exist log\text_matching_question_matching md log\text_matching_question_matching

set logpath=%cd%\log\text_matching_question_matching

cd models_repo\examples\text_matching\question_matching

python train.py --train_set train.txt --dev_set dev.txt --eval_step 10 --save_dir ./checkpoints --train_batch_size 4 --learning_rate 2E-5 -epochs 1 --save_step 10 --max_steps 30 --rdrop_coef 0.0 --device %1  > %logpath%\train_%1.log 2>&1
