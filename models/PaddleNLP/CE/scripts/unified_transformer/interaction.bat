
@echo off
cd ../..

xcopy /y /c /h /r .\scripts\unified_transformer\input.txt  .\models_repo\examples\dialogue\unified_transformer\

if not exist log\unified_transformer md log\unified_transformer

set logpath=%cd%\log\unified_transformer

cd models_repo\examples\dialogue\unified_transformer\

python interaction.py --model_name_or_path=plato-mini --min_dec_len=1 --max_dec_len=64 --num_return_sequences=20 --decode_strategy=sampling --top_k=5 --device=%1 <input.txt > %logpath%\interaction_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%\interaction_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%\interaction_%1.log
)
