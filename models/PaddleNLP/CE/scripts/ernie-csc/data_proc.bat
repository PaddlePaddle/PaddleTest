@echo off
cd ../..

if not exist log\ernie-csc md log\ernie-csc

set logpath=%cd%\log\ernie-csc

cd models_repo\examples\text_correction\ernie-csc\

python download.py --data_dir ./extra_train_ds/ --url https://github.com/wdimmy/Automatic-Corpus-Generation/raw/master/corpus/train.sgml

python change_sgml_to_txt.py -i extra_train_ds/train.sgml -o extra_train_ds/train.txt

python download.py

