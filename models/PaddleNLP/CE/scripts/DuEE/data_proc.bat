@echo off
cd ../..

cd models_repo\examples\information_extraction\DuEE\

rd /s /q data\DuEE-Fin
md data\DuEE-Fin
rd /s /q conf\DuEE-Fin
md conf\DuEE-Fin


xcopy /y /c /h /r D:\ce_data\paddleNLP\DuEE\*  .\data\DuEE-Fin\
xcopy /y /c /h /r D:\ce_data\paddleNLP\DuEE\enum\*  .\data\DuEE-Fin\enum\
xcopy /y /c /h /r D:\ce_data\paddleNLP\DuEE\role\*  .\data\DuEE-Fin\role\
xcopy /y /c /h /r D:\ce_data\paddleNLP\DuEE\sentence\*  .\data\DuEE-Fin\sentence\
xcopy /y /c /h /r D:\ce_data\paddleNLP\DuEE\trigger\*  .\data\DuEE-Fin\trigger\

xcopy /y /c /h /r D:\ce_data\paddleNLP\DuEE\* .\conf\DuEE-Fin\
xcopy /y /c /h /r D:\ce_data\paddleNLP\DuEE\enum\*  .\conf\DuEE-Fin\enum\
xcopy /y /c /h /r D:\ce_data\paddleNLP\DuEE\role\*  .\conf\DuEE-Fin\role\
xcopy /y /c /h /r D:\ce_data\paddleNLP\DuEE\sentence\*  .\conf\DuEE-Fin\sentence\
xcopy /y /c /h /r D:\ce_data\paddleNLP\DuEE\trigger\*  .\conf\DuEE-Fin\trigger\
