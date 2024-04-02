@echo off
cd ../..

if not exist log\seq2seq md log\seq2seq

set logpath=%cd%\log\seq2seq

cd models_repo\examples\machine_translation\seq2seq\

python train.py --num_layers 2 --hidden_size 512 --batch_size 128 --max_epoch 1 --dropout 0.2 --init_scale  0.1 --max_grad_norm 5.0 --device %1 --model_path ./attention_models  > %logpath%\train_%1.log 2>&1
