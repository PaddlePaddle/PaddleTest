@echo off
cd ../..

cd models_repo\examples\model_compression\distill_lstm\

md SST-2
python -m wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt
xcopy /e /y /c /h /r D:\ce_data\paddleNLP\distill_lstm\sst-2\*  .\SST-2\
