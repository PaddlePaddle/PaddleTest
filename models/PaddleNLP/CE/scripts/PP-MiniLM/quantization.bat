@echo off
cd ../..

if not exist log\PP-MiniLM md log\PP-MiniLM
set logpath=%cd%\log\PP-MiniLM
set output_path=%cd%\models_repo\examples\model_compression\pp-minilm\pruning\pruned_models

cd models_repo\examples\model_compression\pp-minilm\quantization

python quant_post.py --task_name %3 --input_dir %output_path%/%3/0.75/sub_static > %logpath%/quantization_%3_%4_%5_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/quantization_%3_%4_%5_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/quantization_%3_%4_%5_%1.log
)
