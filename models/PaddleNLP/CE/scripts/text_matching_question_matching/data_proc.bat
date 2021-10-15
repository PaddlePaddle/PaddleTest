@echo off
cd ../..

cd models_repo\examples\text_matching\question_matching

xcopy /e /y /c /h /r D:\ce_data\paddleNLP\question_matching\data_v4\  .
