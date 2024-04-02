@echo off
cd ../..

if not exist log\seq2seq md log\seq2seq

set logpath=%cd%\log\seq2seq

cd models_repo\examples\machine_translation\seq2seq\

python export_model.py --num_layers 2 --hidden_size 512 --batch_size 128 --dropout 0.2 --init_scale  0.1 --max_grad_norm 5.0 --init_from_ckpt attention_models/final.pdparams --beam_size 10 --export_path ./Infer_model/model > %logpath%/inferfram_%1.log 2>&1


if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/inferfram_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/inferfram_%1.log
)
