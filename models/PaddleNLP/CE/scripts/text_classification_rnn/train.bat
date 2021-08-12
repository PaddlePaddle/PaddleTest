@echo off
cd ../..

md log\text_classification_rnn
set logpath=%cd%\log\text_classification_rnn

cd models_repo\examples\text_classification\rnn\

python -m paddle.distributed.launch --gpus %2 train.py --vocab_path=.\senta_word_dict.txt --device=%1 --network=bilstm --lr=5e-4 --batch_size=64 --epochs=1 --save_dir=.\checkpoints > %logpath%/train_%1.log 2>&1
