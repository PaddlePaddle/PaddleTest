@echo off
cd ../..

if not exist log\transformer-xl md log\transformer-xl

set logpath=%cd%\log\transformer-xl

cd models_repo\examples\language_model\transformer-xl\

xcopy /e /y /c /h /r D:\ce_data\paddleNLP\transformer-xl\*  .\

set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -i 's/print_step: 100/print_step: 1/g' configs/enwik8.yaml
%sed% -i 's/batch_size: 16/batch_size: 8/g' configs/enwik8.yaml
%sed% -i 's/save_step: 10000/save_step: 3/g' configs/enwik8.yaml
%sed% -i 's/max_step: 400000/max_step: 4/g' configs/enwik8.yaml

python -m paddle.distributed.launch  --gpus %2 train.py --config ./configs/enwik8.yaml > %logpath%\enwik8_train_%1.log 2>&1
