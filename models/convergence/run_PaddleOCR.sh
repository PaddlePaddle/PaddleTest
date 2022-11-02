export http_proxy=http://172.19.57.45:3128
export https_proxy=http://172.19.57.45:3128
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
python -m pip install -r requirements.txt
ln -s /paddle/data/ce_data/PaddleOCR/train_data train_data
ln -s /paddle/data/ce_data/PaddleOCR/pretrain_models pretrain_models
# db_4card
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir log_db_mv3
python -m paddle.distributed.launch --log_dir=log_db_mv3 --gpus '0,1,2,3'  tools/train.py -c configs/det/det_mv3_db.yml > log_db_mv3/det_mv3_db.log 2>&1 &
# rec_4card
export CUDA_VISIBLE_DEVICES=4,5,6,7
mkdir log_rec_mv3_none_bilstm_ctc
python -m paddle.distributed.launch --log_dir=rec_mv3_none_bilstm_ctc --gpus '4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml> log_rec_mv3_none_bilstm_ctc/rec_mv3_none_bilstm_ctc.log 2>&1 &
