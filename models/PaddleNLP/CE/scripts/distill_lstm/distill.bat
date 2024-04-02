@echo off
cd ../..

if not exist log\distill_lstm md log\distill_lstm
set logpath=%cd%\log\distill_lstm

cd models_repo\examples\model_compression\distill_lstm\
if "%2"=="sst-2" (
python bert_distill.py --task_name sst-2 --vocab_size 30522 --max_epoch 1 --lr 1.0 --task_name sst-2 --dropout_prob 0.2 --batch_size 128 --model_name bert-base-uncased --output_dir distilled_models/SST-2 --teacher_dir ./SST-2/best_model_610 --save_steps 1000 --device %1 --embedding_name w2v.google_news.target.word-word.dim300.en > %logpath%/distill_%2_%1.log 2>&1
)
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/distill_%2_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/distill_%2_%1.log
)
