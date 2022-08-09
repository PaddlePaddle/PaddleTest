@echo off
cd ../..

cd models_repo\examples\model_compression\pp-minilm\general_distill

xcopy /e /y /c /h /r D:\ce_data\paddleNLP\PP-MiniLM\*  .
