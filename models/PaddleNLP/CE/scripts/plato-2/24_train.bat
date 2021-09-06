@echo off
cd ../..

xcopy /y /c /h /r .\scripts\plato-2\input.txt  .\models_repo\examples\dialogue\plato-2\

if not exist log\plato-2 md log\plato-2

set logpath=%cd%\log\plato-2

cd models_repo\examples\dialogue\plato-2\


python interaction.py --vocab_path ./data/vocab.txt --spm_model_file ./data/spm.model --num_layers 24 --init_from_ckpt ./24L.pdparams < input.txt  > %logpath%/train_24_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_24_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_24_%1.log
)
