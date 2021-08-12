@echo off
cd ../..

md log\skep

set logpath=%cd%\log\skep

cd models_repo\examples\sentiment_analysis\skep\


python -m paddle.distributed.launch --gpus %2 train_sentence.py --model_name "skep_ernie_1.0_large_ch" --device %1 --epochs 1 --save_dir ./checkpoints > %logpath%/train_%1.log 2>&1
