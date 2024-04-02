@echo off
cd ../..

set logpath=%cd%\log\msra_ner

cd models_repo\examples\information_extraction\msra_ner\

python -u ./eval.py --model_name_or_path bert-base-multilingual-uncased --max_seq_length 128 --batch_size 4 --device %1 --init_checkpoint_path tmp/msra_ner/model_500.pdparams > %logpath%/eval_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/eval_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/eval_%1.log
)
