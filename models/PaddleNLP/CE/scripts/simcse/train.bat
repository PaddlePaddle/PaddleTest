@echo off
cd ../..
if not exist log\simcse md log\simcse

set logpath=%cd%\log\simcse

cd models_repo\examples\text_matching\simcse\

python -u -m paddle.distributed.launch --gpus %2 train.py --save_dir ./%3 --batch_size 16 --learning_rate 5E-5 --epochs 1 --save_steps 20 --eval_steps 100 --max_steps 100 --max_seq_length 64 --infer_with_fc_pooler --dropout 0.1 --train_set_file "./senteval_cn/%3/train.txt" --test_set_file "./senteval_cn/%3/dev.tsv" --device %1 > %logpath%/train_%3_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%4_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%4_%1.log
)
