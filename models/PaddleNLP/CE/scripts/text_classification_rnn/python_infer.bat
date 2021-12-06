@echo off
cd ../..

set logpath=%cd%\log\text_classification_rnn

cd models_repo\examples\text_classification\rnn\

python export_model.py --vocab_path=./senta_word_dict.txt --network=bilstm --params_path=./checkpoints/final.pdparams --output_path=./static_graph_params

python deploy/python/predict.py --model_file=static_graph_params.pdmodel --params_file=static_graph_params.pdiparams --network=bilstm --device=%1 > %logpath%/python_infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/python_infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/python_infer_%1.log
)
