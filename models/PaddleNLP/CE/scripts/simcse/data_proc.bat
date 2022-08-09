@echo off
cd ../..

cd models_repo\examples\text_matching\simcse\

xcopy /e /y /c /h /r D:\ce_data\paddleNLP\simcse\*  .\

cd .\senteval_cn

move BQ BQ_Corpus
