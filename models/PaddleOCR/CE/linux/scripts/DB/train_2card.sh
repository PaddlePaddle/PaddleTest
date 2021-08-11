export FLAGS_cudnn_deterministic=True
cd /workspace/PaddleOCR/ce/Paddle_Cloud_CE/src/task/PaddleOCR

if [ ! -f "pretrain_models/MobileNetV3_large_x0_5_pretrained.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams
fi

rm -rf train_data
ln -s /home/data/cfs/models_ce/PaddleOCR/train_data train_data

if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt

python -m paddle.distributed.launch tools/train.py -c configs/det/det_mv3_db.yml -o Global.epoch_num=10 > log/det_mv3_db_2card.log 2>&1
cat log/det_mv3_db_2card.log | grep "10/10" > ../log/det_mv3_db_2card_tmp.log

linenum=`cat ../log/det_mv3_db_2card_tmp.log | wc -l`
linenum_last1=`expr $linenum - 1`
if [ $linenum_last1 -eq 0 ] 
  then cp ../log/det_mv3_db_2card_tmp.log ../log/det_mv3_db_2card.log
  else sed ''1,"$linenum_last1"'d' ../log/det_mv3_db_2card_tmp.log > ../log/det_mv3_db_2card.log
fi
rm -rf ../log/det_mv3_db_2card_tmp.log

