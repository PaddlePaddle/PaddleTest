sed -i '' 's!batch_size_per_card: 512!batch_size_per_card: 16!g' configs/cls/cls_mv3.yml
sed -ie '/- RecAug:/{N;d;}' configs/cls/cls_mv3.yml
sed -ie '/- RandAugment:/d' configs/cls/cls_mv3.yml #删除 RandAugment 字段行

sed -i '' 's!data_lmdb_release/training!data_lmdb_release/validation!g' configs/rec/rec_mv3_none_bilstm_ctc.yml

if [ ! -f "pretrain_models/MobileNetV3_large_x0_5_pretrained.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams
fi

if [ ! -f "pretrain_models/en_server_pgnetA/best_accuracy.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar
  cd pretrain_models
  tar -xf en_server_pgnetA.tar
  cd ../
fi

# sed -i '' 's/scikit-image==0.17.2/scikit-image/g' requirements.txt

# data
rm -rf train_data
ln -s $1/PaddleOCR/train_data train_data
