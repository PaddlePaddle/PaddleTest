ln -s  /ssd2/ce_data/PaddleOCR/train_data train_data
python -m pip install -r requirements.txt
export no_proxy=.bcebos.com
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams
rm -rf log_det
mkdir log_det
python -m paddle.distributed.launch --log_dir=./log_det/ --gpus '0,1,2,3'  tools/train.py -c configs/det/det_mv3_db.yml > log_det/det_mv3_db.log 2>&1 &
