@echo off
cd ../..

md log\stacl
set logpath=%cd%\log\stacl

cd models_repo\examples\simultaneous_translation\stacl\

set sed="C:\Program Files\Git\usr\bin\sed.exe"

%sed% -i "s/save_step: 10000/save_step: 10/g" config/transformer.yaml
%sed% -i "s/print_step: 100/print_step: 10/g" config/transformer.yaml
%sed% -i "s/max_iter: None/max_iter: 30/g" config/transformer.yaml
%sed% -i "s/epoch: 30/epoch: 1/g" config/transformer.yaml
%sed% -i "s/batch_size: 4096/batch_size: 100/g" config/transformer.yaml
%sed% -i 's/init_from_params: \"trained_models\/step_final\/\"/init_from_params: \"trained_models\/step_10\/\"/g' config/transformer.yaml

python -m paddle.distributed.launch --gpus %2 train.py --config ./config/transformer.yaml > %logpath%/train_%1.log 2>&1
