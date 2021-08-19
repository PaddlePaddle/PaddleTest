@echo off
cd ../..

md log\msra_ner
set logpath=%cd%\log\msra_ner

cd models_repo\examples\information_extraction\msra_ner\

python -u ./train.py --model_name_or_path bert-base-multilingual-uncased --max_seq_length 128 --batch_size 32 --learning_rate 2e-5 --num_train_epochs 1 --logging_steps 10 --save_steps 500 --max_steps 1000 --output_dir ./tmp/msra_ner/ --device %1 > %logpath%/train_%1.log 2>&1
