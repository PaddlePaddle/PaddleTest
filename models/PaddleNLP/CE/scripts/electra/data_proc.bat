@echo off
cd ../..

cd models_repo

cd examples\language_model\electra

md BookCorpus

xcopy /y /c /h /r D:\ce_data\paddleNLP\electra\train.data   .\BookCorpus\
