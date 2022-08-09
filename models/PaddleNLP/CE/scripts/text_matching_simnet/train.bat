@echo off
cd ../..
if not exist log\text_matching_simnet md log\text_matching_simnet
set logpath=%cd%\log\text_matching_simnet
cd models_repo\examples\text_matching\simnet\
python -m paddle.distributed.launch --gpus %2 train.py --vocab_path="./simnet_vocab.txt" --device=%1 --network=lstm --lr=5e-4 --batch_size=64 --epochs=1 --save_dir="./checkpoints" > %logpath%/train_%1.log 2>&1
