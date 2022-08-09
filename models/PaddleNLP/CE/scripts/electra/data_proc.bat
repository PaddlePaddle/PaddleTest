@echo off
cd ../..

cd models_repo

cd model_zoo\electra

md BookCorpus

xcopy /y /c /h /r D:\ce_data\paddleNLP\electra\train.data   .\BookCorpus\
