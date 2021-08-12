@echo off
cd ../..
cd models_repo\examples\simultaneous_translation\stacl\
python -m pip install wget
python -m pip install -r requirements.txt
rd /s /q data\nist2m
md data\nist2m

xcopy /y /c /h /r D:\ce_data\paddleNLP\stacl\*  .\data\nist2m\
