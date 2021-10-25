export FLAGS_cudnn_deterministic=True
cd ${Project_path}

if [ ! -f "pretrain_models/MobileNetV3_large_x0_5_pretrained.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams
fi

if [ ! -f "pretrain_models/ResNet18_vd_pretrained.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_vd_pretrained.pdparams
fi

if [ ! -f "pretrain_models/ResNet50_vd_ssld_pretrained.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams
fi

if [ ! -f "pretrain_models/ResNet50_vd_pretrained.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams
fi

if [ ! -f "pretrain_models/ch_ppocr_mobile_v2.0_det_train.tar" ]; then
    wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
    tar xf pretrain_models/ch_ppocr_mobile_v2.0_det_train.tar -C pretrain_models
fi

if [ ! -f "pretrain_models/ch_ppocr_server_v2.0_det_train.tar" ]; then
    wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar
    tar xf pretrain_models/ch_ppocr_server_v2.0_det_train.tar -C pretrain_models
fi

rm -rf train_data
ln -s ${Data_path}/train_data train_data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt

python -m paddle.distributed.launch tools/train.py -c configs/det/det_r50_vd_sast_icdar15.yml -o Global.epoch_num=2 > log/det_r50_vd_sast_icdar15_2card.log 2>&1
cat log/det_r50_vd_sast_icdar15_2card.log | grep "2/2" > ../log/det_r50_vd_sast_icdar15_2card_tmp.log

linenum=`cat ../log/det_r50_vd_sast_icdar15_2card_tmp.log | wc -l`
linenum_last1=`expr $linenum - 1`
if [ $linenum_last1 -eq 0 ]
  then cp ../log/det_r50_vd_sast_icdar15_2card_tmp.log ../log/det_r50_vd_sast_icdar15_2card.log
  else sed ''1,"$linenum_last1"'d' ../log/det_r50_vd_sast_icdar15_2card_tmp.log > ../log/det_r50_vd_sast_icdar15_2card.log
fi
rm -rf ../log/det_r50_vd_sast_icdar15_2card_tmp.log
