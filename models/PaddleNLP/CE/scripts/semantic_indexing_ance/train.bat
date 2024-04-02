@echo off
cd ../..

md log\semantic_indexing_ance
set logpath=%cd%\log\semantic_indexing_ance

cd models_repo\examples\semantic_indexing\

if "%3"=="batch" (
    python -u -m paddle.distributed.launch --gpus %2 train_batch_neg.py --device %1 --save_dir ./checkpoints_batch_neg/ --batch_size 32 --learning_rate 5E-5 --epochs 1 --output_emb_size 256 --save_steps 1000 --max_seq_length 64 --margin 0.2 --train_set_file semantic_pair_train.tsv > %logpath%\train_%3_%1.log 2>&1
) else (
    python -u -m paddle.distributed.launch --gpus %2 train_hardest_neg.py --device %1 --save_dir ./checkpoints_hardest_neg/ --batch_size 32 --learning_rate 5E-5 --epochs 1 --output_emb_size 256 --save_steps 1000 --max_seq_length 64 --margin 0.2 --train_set_file semantic_pair_train.tsv > %logpath%\train_%3_%1.log 2>&1
)
