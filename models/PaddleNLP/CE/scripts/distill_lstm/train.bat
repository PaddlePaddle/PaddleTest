@echo off
cd ../..

if not exist log\distill_lstm md log\distill_lstm
set logpath=%cd%\log\distill_lstm
set model_path=C:\Users\paddle-ci\.paddlenlp\models\bert-base-uncased\bert-base-uncased-vocab.txt

cd models_repo\examples\model_compression\distill_lstm\
if "%2"=="sst-2" (
python small.py --task_name sst-2 --vocab_size 30522 --max_epoch 1 --batch_size 8 --lr 1.0 --dropout_prob 0.4 --output_dir small_models/SST-2 --save_steps 1000 --vocab_path %model_path% --device %1 --embedding_name w2v.google_news.target.word-word.dim300.en > %logpath%/train_%2_%1.log 2>&1
)
