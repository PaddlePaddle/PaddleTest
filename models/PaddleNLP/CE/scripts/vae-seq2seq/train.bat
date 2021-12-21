@echo off
cd ../..

if not exist log\vae-seq2seq md log\vae-seq2seq

set logpath=%cd%\log\vae-seq2seq

cd models_repo\examples\text_generation\vae-seq2seq\

python train.py --batch_size 32 --init_scale 0.1 --max_grad_norm 5.0 --dataset ptb --model_path ptb_model --device %1 --max_epoch 1 > %logpath%/train_%1.log 2>&1


