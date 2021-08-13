@echo off
cd ../..

set logpath=%cd%\log\text_classification_pretrained

cd models_repo\examples\text_classification\pretrained_models\


python predict.py --device %1 --params_path checkpoints/model_900/model_state.pdparams > %logpath%/infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
