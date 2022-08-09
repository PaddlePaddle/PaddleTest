@echo off
cd ../..

if not exist log\vae-seq2seq md log\vae-seq2seq

set logpath=%cd%\log\vae-seq2seq

cd models_repo\examples\text_generation\vae-seq2seq\

python predict.py --batch_size 32 --init_scale 0.1 --max_grad_norm 5.0 --dataset ptb --device %1 --infer_output_file infer_output.txt --init_from_ckpt ptb_model/final > %logpath%\infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%\infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%\infer_%1.log
)
