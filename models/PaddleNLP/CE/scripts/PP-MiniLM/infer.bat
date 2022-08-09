@echo off
cd ../..

if not exist log\PP-MiniLM md log\PP-MiniLM
set logpath=%cd%\log\PP-MiniLM
cd models_repo\examples\model_compression\pp-minilm\deploy\python

set NAME=%3

for %%i in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do call set NAME=%%NAME:%%i=%%i%%

python infer.py --task_name %NAME% --model_path  ../../quantization/%NAME%_quant_models/mse4/int8  --int8 --collect_shape
python infer.py --task_name %NAME%  --model_path  ../../quantization/%NAME%_quant_models/mse4/int8  --int8 > %logpath%/infer_%3_%4_%5_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%3_%4_%5_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%3_%4_%5_%1.log
)
