@echo off
cd ../..

if not exist log\bigbird md log\bigbird

cd models_repo\examples\language_model\bigbird\

if not exist data md data

cd data

python -m wget https://paddlenlp.bj.bcebos.com/ce/data/bigbird/wiki.csv