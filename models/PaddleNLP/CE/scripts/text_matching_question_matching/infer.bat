@echo off
cd ../..

if not exist log\text_matching_question_matching md log\text_matching_question_matching

set logpath=%cd%\log\text_matching_question_matching

cd models_repo\examples\text_matching\question_matching

python -u predict.py --device %1 --params_path "./checkpoints/model_20/model_state.pdparams" --batch_size 16 --input_file ./test/public_test_A --result_file "predict_result" > %logpath%\infer_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/infer_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/infer_%1.log
)
