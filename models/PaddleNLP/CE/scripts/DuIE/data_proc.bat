@echo off
cd ../..

cd models_repo\examples\information_extraction\DuIE\

xcopy /y /c /h /r D:\ce_data\paddleNLP\DuIE\*  .\data\

set sed="C:\Program Files\Git\usr\bin\sed.exe"

%sed% -i "s/python3 .\/re_official_evaluation.py/python .\/re_official_evaluation.py/g"  ./utils.py
