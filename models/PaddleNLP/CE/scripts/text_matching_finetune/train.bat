@echo off
cd ../..

md log\text_matching_finetune
set logpath=%cd%\log\text_matching_finetune

cd models_repo\examples\text_matching\sentence_transformers\

set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -i "s/if global_step % 100 == 0 and rank == 0:/if global_step % 1000 == 0 and rank == 0:/g" train.py

python -m paddle.distributed.launch --gpus %2 train.py --device %1 --save_dir ./checkpoints  --epochs 1 > %logpath%/train_%1.log 2>&1
