rm -rf log_rec
mkdir log_rec
python -m paddle.distributed.launch --log_dir=./log_rec/ --gpus '4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml > log_rec/rec_mv3_none_bilstm_ctc.log 2>&1 &
