@echo off
cd ../..

if not exist log\ofa md log\ofa

set logpath=%cd%\log\ofa

cd models_repo\examples\model_compression\ofa\

xcopy /e /y /c /h /r D:\ce_data\paddleNLP\ofa\  .\

python -m paddle.distributed.launch run_glue_ofa.py  --model_type bert --model_name_or_path=./sst-2_ft_model_1.pdparams  --task_name SST-2 --max_seq_length 128  --batch_size 16  --learning_rate 2e-5  --num_train_epochs 1  --max_steps 1 --logging_steps 1   --save_steps 1   --output_dir ./ofa/SST-2  --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5 --device=%1  > %logpath%\train_ofa_%1.log 2>&1
