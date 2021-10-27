@echo off
cd ../..

if not exist log\unimo-text md log\unimo-text

set logpath=%cd%\log\unimo-text

cd models_repo\examples\text_generation\unimo-text\

python run_gen.py --dataset_name=dureader_qg --model_name_or_path=./unimo/checkpoints/model_3630/ --logging_steps=100 --batch_size=2 --max_seq_len=512 --max_target_len=30 --do_predict --max_dec_len=20 --min_dec_len=3 --device=%1 > %logpath%\infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%\infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%\infer_%1.log
)
