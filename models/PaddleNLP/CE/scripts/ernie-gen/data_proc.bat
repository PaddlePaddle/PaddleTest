@echo off
cd ../..

cd models_repo\examples\text_generation\ernie-gen\

python -m wget https://paddlenlp.bj.bcebos.com/datasets/poetry.tar.gz
tar xvf poetry.tar.gz
move .\poetry\train.tsv  .\poetry\train_origin.tsv
set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -n '1,5p' .\poetry\train_origin.tsv > .\poetry\train.tsv
