@echo off
cd ../..
cd models_repo\examples\text_classification\rnn\
python -m pip install wget
python -m wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt
