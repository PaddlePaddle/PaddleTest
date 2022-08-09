@echo off
cd ../..

if not exist log\applications_sentiment_analysis md log\applications_sentiment_analysis
set logpath=%cd%\log\applications_sentiment_analysis

cd models_repo\applications\sentiment_analysis\classification\

python train.py --train_path "../data/cls_data/train.txt" --dev_path "../data/cls_data/dev.txt" --label_path "../data/cls_data/label.dict" --num_epochs 1 --batch_size 1 --max_seq_len 256 --learning_rate 3e-5 --weight_decay 0.01 --max_grad_norm 1.0 --warmup_proportion 0.1 --log_steps 50 --eval_steps 100 --seed 1000 --device %1 --checkpoints "../checkpoints/cls_checkpoints" > %logpath%\classification_train_%1.log 2>&1
