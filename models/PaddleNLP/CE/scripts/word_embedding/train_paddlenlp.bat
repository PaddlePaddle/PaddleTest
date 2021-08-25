
@echo off
cd ../..

if not exist log\word_embedding md log\word_embedding

set logpath=%cd%\log\word_embedding

cd models_repo\examples\word_embedding\

python train.py --lr=1e-4 --batch_size=32 --epochs=1 --use_token_embedding=False --vdl_dir='./vdl_paddle_dir' --device=%1  > %logpath%\train_paddlenlp_%1.log 2>&1