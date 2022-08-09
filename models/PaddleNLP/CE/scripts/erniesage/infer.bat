@echo off
cd ../..

if not exist log\erniesage md log\erniesage

set logpath=%cd%\log\erniesage

cd models_repo\examples\text_graph\erniesage\


python -m paddle.distributed.launch --gpus %2 link_prediction.py --conf ./config/erniesage_link_prediction.yaml --do_predict > %logpath%/infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
