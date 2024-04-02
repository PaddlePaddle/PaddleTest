@echo off
cd ../..
if not exist log\text_matching_simnet md log\text_matching_simnet
set logpath=%cd%\log\text_matching_simnet
cd models_repo\examples\text_matching\simnet\
python predict.py --vocab_path="./simnet_vocab.txt" --device=%1 --network=lstm --params_path="./checkpoints/final.pdparams" > %logpath%/infer_%1.log 2>&1
if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
