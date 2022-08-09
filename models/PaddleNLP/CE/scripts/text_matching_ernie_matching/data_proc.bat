@echo off
cd ../..

cd models_repo\examples\text_matching\ernie_matching\

xcopy /y /c /h /r C:\Users\paddle-ci\.paddlenlp\datasets\LCQMC\lcqmc\lcqmc\test.tsv .\
