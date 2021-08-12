@echo off
cd ../..

set logpath=%cd%\log\text_classification_rnn

cd models_repo\examples\text_classification\rnn\

python predict.py --vocab_path=.\senta_word_dict.txt --device=%1 --network=bilstm --params_path=.\checkpoints\final.pdparams > %logpath%/infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
