@echo off
cd ../..

set logpath=%cd%\log\DuEE

cd models_repo\examples\information_extraction\DuEE\

set sed="C:\Program Files\Git\usr\bin\sed.exe"

%sed% -i "s/with open(path, \"w\") as outfile:/with open(path, \"w\", encoding='utf-8') as outfile:/g" utils.py

if "%2"=="enum" (
    python classifier.py --num_epoch 1 --learning_rate 5e-5 --tag_path ./conf/DuEE-Fin/%2_tag.dict --train_data ./data/DuEE-Fin/%2/train.tsv --dev_data ./data/DuEE-Fin/%2/dev.tsv --test_data ./data/DuEE-Fin/%2/test.tsv --predict_data ./data/DuEE-Fin/sentence/test.json --do_train False --do_predict True --max_seq_len 300 --batch_size 16 --skip_step 1 --valid_step 1 --checkpoints ./ckpt/DuEE-Fin/%2 --init_ckpt ./ckpt/DuEE-Fin/%2/final.pdparams --predict_save_path ./ckpt/DuEE-Fin/%2/test_pred.json --device %1 > %logpath%/infer_%2_%1.log 2>&1
) else (
    python sequence_labeling.py --num_epoch 1 --learning_rate 5e-5 --tag_path ./conf/DuEE-Fin/%2_tag.dict --train_data ./data/DuEE-Fin/%2/train.tsv --dev_data ./data/DuEE-Fin/%2/dev.tsv --test_data ./data/DuEE-Fin/%2/test.tsv --predict_data ./data/DuEE-Fin/sentence/test.json --do_train False --do_predict True --max_seq_len 300 --batch_size 1 --skip_step 10 --valid_step 50 --checkpoints ./ckpt/DuEE-Fin/%2 --init_ckpt ./ckpt/DuEE-Fin/%2/best.pdparams --predict_save_path ./ckpt/DuEE-Fin/%2/test_pred.json --device %1 > %logpath%/infer_%2_%1.log 2>&1
)

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%2_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%2_%1.log
)
