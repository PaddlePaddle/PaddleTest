@echo off
cd ../..

set logpath=%cd%\log\msra_ner

cd models_repo\examples\information_extraction\msra_ner\

python -u ./predict.py --model_name_or_path bert-base-multilingual-uncased --max_seq_length 128 --batch_size 32 --device %1 --init_checkpoint_path tmp/msra_ner/model_500.pdparams > %logpath%/infer_%1.log 2>&1
